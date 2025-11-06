import os.path as osp
import logging
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import pytorch_lightning as pl

from src.utils.bbox_utils import CropResizePad
from src.model.utils import BatchedData

from typing_extensions import Union, Tuple, Optional


descriptor_map = {
    "dinov2_vits14": "vit_small",
    "dinov2_vitb14": "vit_base",
    "dinov2_vitl14": "vit_large",
    "dinov2_vitg14": "vit_giant2",
}


class CustomDINOv2(pl.LightningModule):
    def __init__(self,
                 model_name: str,
                 model_or_repo: Union[str, object],
                 image_size: int,
                 chunk_size: int,
                 patch_size: int = 14,
                 valid_patch_mask_threshold: float = 0.5):
        super().__init__()
        self.model_name = model_name

        if isinstance(model_or_repo, str):
            # load model from repo dir
            self.model = torch.hub.load(repo_or_dir=model_or_repo, model=model_name, force_reload=False)
            self.model.eval()
        else:
            # model already loaded
            self.model = model_or_repo

        self.valid_patch_mask_threshold = valid_patch_mask_threshold
        self.chunk_size = chunk_size
        self.patch_size = patch_size
        self.image_size = image_size

        # pre processing
        self.rgb_normalize = T.Compose([T.ToTensor(),
                                        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        # use for global feature
        self.rgb_proposal_processor = CropResizePad(self.image_size)
        self.patch_kernel = torch.nn.AvgPool2d(kernel_size=self.patch_size, stride=self.patch_size)

        logging.info("Init CustomDINOv2 with image size={} done!".format(self.image_size))

    def process_rgb_proposals(self, image: np.ndarray, masks: Optional[torch.Tensor], boxes: torch.Tensor) -> torch.Tensor:
        """
        1. Normalize image
        2. Mask and crop each proposals
        3. Resize each proposal to the predefined image size

        :param np.ndarray image: [H, W, C]
        :param Optional[torch.Tensor] masks: [B, H, W]
        :param torch.Tensor boxes: [B, 4] (xywh)
        :return: processed image crops [B, C, S, S]
        :rtype: torch.Tensor
        """
        rgb = self.rgb_normalize(image).float()    # C x H x W
        if masks is None:
            # no mask available, so use the number of bounding boxes
            masked_rgbs = rgb.unsqueeze(0).repeat(boxes.shape[0], 1, 1, 1).to(boxes.device)             # B x C x H x W
        else:
            # apply all masks to the image
            masked_rgbs = rgb.unsqueeze(0).to(boxes.device) * masks.unsqueeze(1).to(boxes.device)       # B x C x H x W

        processed_masked_rgbs = self.rgb_proposal_processor(masked_rgbs, boxes)     # B x C x S x S

        return processed_masked_rgbs.to(boxes.device)   # B x C x S x S

    @torch.no_grad()
    def compute_features(self, processed_rgbs: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        """
        :param torch.Tensor processed_rgbs: [B, C, S, S]
        :param bool normalize:
        :return: cls tokens [B, cF]
        :rtype: torch.Tensor
        """
        if processed_rgbs.shape[0] > self.chunk_size:
            features = self.forward_by_chunk(processed_rgbs, normalize)     # B x cF
        else:
            features = self.model(processed_rgbs)               # chunk_size x cF

            if normalize:
                # normalize
                features = F.normalize(features, dim=-1)        # chunk_size x cF

        return features     # (B|chunk_size) x cF

    @torch.no_grad()
    def forward_by_chunk(self, processed_rgbs: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        """
        :param torch.Tensor processed_rgbs: [B, C, S, S]
        :param bool normalize:
        :return: cls tokens [B, cF]
        :rtype: torch.Tensor
        """
        batch_rgbs = BatchedData(batch_size=self.chunk_size, data=processed_rgbs)
        del processed_rgbs  # free memory

        features = [None] * len(batch_rgbs)
        for index_batch in range(len(batch_rgbs)):
            features[index_batch] = self.compute_features(batch_rgbs[index_batch], normalize)   # chunk_size x cF
        features = torch.cat(features, dim=0)   # B x cF

        return features     # B x cF

    @torch.no_grad()
    def forward_cls_token(self, image: np.ndarray, proposals,
                          mask_rgbs: bool = True,
                          normalize: bool = False) -> torch.Tensor:
        """
        :param np.ndarray image: [H, W, C]
        :param proposals: object with variables: {"boxes": B x 4 (as xyxy), "masks": B x H x W}
        :param bool mask_rgbs:
        :param bool normalize:
        :return: cls tokens [B, cF]
        :rtype: torch.Tensor
        """
        processed_rgbs = self.process_rgb_proposals(image, proposals.masks if mask_rgbs else None,
                                                    proposals.boxes)    # B x C x S x S
        return self.forward_by_chunk(processed_rgbs, normalize)         # B x cF

    def process_masks_proposals(self, masks: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """
        Crop and resize each mask to the predefined image size

        :param torch.Tensor masks: [B, H, W]
        :param torch.Tensor boxes: [B, 4]   (xywh)
        :return: processed mask crops [B, S, S]
        :rtype: torch.Tensor
        """
        masks = masks.clone().unsqueeze_(1)     # B x 1 x H x W
        processed_masks = self.rgb_proposal_processor(masks, boxes).squeeze_(1)     # B x S x S

        return processed_masks  # B x S x S

    @torch.no_grad()
    def compute_masked_patch_feature(self, processed_rgbs: torch.Tensor, masks: torch.Tensor, normalize: bool = False)\
            -> torch.Tensor:
        """
        :param torch.Tensor images: [B, C, S, S]
        :param torch.Tensor masks: [B, S, S]
        :param bool normalize:
        :return: patch tokens [B, P*P, pF]
        :rtype: torch.Tensor
        """
        # without preprocess
        if processed_rgbs.shape[0] > self.chunk_size:
            features = self.forward_by_chunk_patch_feature(processed_rgbs, masks, normalize)    # B x (P*P) x pF
        else:
            features = self.model(processed_rgbs, is_training=True)["x_norm_patchtokens"]   # chunk_size x (P*P) x pF

            # determine the mask for feature patch embeddings
            features_mask = self.patch_kernel(masks).flatten(start_dim=-2) > self.valid_patch_mask_threshold    # chunk_size x (P*P)

            # mask embeddings
            features = features * features_mask.unsqueeze(-1).to(features.device)   # chunk_size x (P*P) x pF

            if normalize:
                # normalize
                features = F.normalize(features, dim=-1)    # chunk_size x (P*P) x pF

        return features     # (B|chunk_size) x (P*P) x pF

    @torch.no_grad()
    def forward_by_chunk_patch_feature(self, processed_rgbs: torch.Tensor, masks: torch.Tensor, normalize: bool = False)\
            -> torch.Tensor:
        """"
        :param torch.Tensor processed_rgbs: [B, C, S, S]
        :param torch.Tensor masks: [B, S, S]
        :param bool normalize:
        :return: patch tokens [B, P*P, pF]
        :rtype: torch.Tensor
        """
        batch_rgbs = BatchedData(batch_size=self.chunk_size, data=processed_rgbs)
        batch_masks = BatchedData(batch_size=self.chunk_size, data=masks)
        del processed_rgbs  # free memory
        del masks           # free memory

        features = [None] * len(batch_rgbs)
        for index_batch in range(len(batch_rgbs)):
            features[index_batch] = self.compute_masked_patch_feature(batch_rgbs[index_batch],
                                                                      batch_masks[index_batch],
                                                                      normalize)    # chunk_size x (P*P) x pF
        features = torch.cat(features, dim=0)   # B x (P*P) x pF

        return features     # B x (P*P) x pF

    @torch.no_grad()
    def forward_patch_tokens(self, image: np.ndarray, proposals,
                             mask_rgbs: bool = True,
                             normalize: bool = False) -> torch.Tensor:
        """
        :param  np.ndarray image: [H, W, C]
        :param proposals: object with variables: {"boxes": B x 4 (as xyxy), "masks": B x H x W}
        :param bool mask_rgbs:
        :param bool normalize:
        :return: patch tokens [B, P*P, pF]
        :rtype: torch.Tensor
        """
        # preprocess image and masks
        processed_rgbs = self.process_rgb_proposals(image, proposals.masks if mask_rgbs else None,
                                                    proposals.boxes)                        # B x C x S x S
        processed_masks = self.process_masks_proposals(proposals.masks, proposals.boxes)    # B x S x S

        return self.forward_by_chunk_patch_feature(processed_rgbs, processed_masks, normalize)     # B x (P*P) x pF

    def compute_cls_and_patch_features(self, processed_rgbs: torch.Tensor, masks: torch.Tensor,
                                       normalize_cls: bool = False, normalize_patch: bool = False)\
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param torch.Tensor processed_rgbs: [B, C, S, S]
        :param torch.Tensor masks: [B, S, S]
        :param bool normalize_cls:
        :param bool normalize_patch:
        :return: A tuple containing the cls tokens [B, cF] and patch tokens [B, P*P, pF]
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        if processed_rgbs.shape[0] > self.chunk_size:
            cls_features, patch_features = self.forward_by_chunk_cls_and_patch_features(
                processed_rgbs,
                masks,
                normalize_cls,
                normalize_patch)  # B x cF,  B x (P*P) x pF
        else:
            features = self.model(processed_rgbs, is_training=True)
            patch_features = features["x_norm_patchtokens"]     # chunk_size x (P*P) x pF
            cls_features = features["x_norm_clstoken"]          # chunk_size x cF

            # determine the mask for feature patch embeddings
            features_mask = self.patch_kernel(masks).flatten(start_dim=-2) > self.valid_patch_mask_threshold    # chunk_size x (P*P)

            # mask embeddings
            patch_features = patch_features * features_mask.unsqueeze(-1).to(patch_features.device)     # chunk_size x (P*P) x pF

            if normalize_cls:
                # normalize cls features
                cls_features = F.normalize(cls_features, dim=-1)        # chunk_size x pF
            if normalize_patch:
                # normalize patch features
                patch_features = F.normalize(patch_features, dim=-1)    # chunk_size x (P*P) x pF

        return cls_features, patch_features     # (B|chunk_size) x cF, (B|chunk_size) x (P*P) x pF

    def forward_by_chunk_cls_and_patch_features(self, processed_rgbs: torch.Tensor, masks: torch.Tensor,
                                                normalize_cls: bool = False, normalize_patch: bool = False)\
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param torch.Tensor processed_rgbs: [B, C, S, S]
        :param torch.Tensor masks: [B, S, S]
        :param bool normalize_cls:
        :param bool normalize_patch:
        :return: A tuple containing the cls tokens [B, cF] and patch tokens [B, P*P, pF]
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        batch_rgbs = BatchedData(batch_size=self.chunk_size, data=processed_rgbs)
        batch_masks = BatchedData(batch_size=self.chunk_size, data=masks)
        del processed_rgbs  # free memory
        del masks           # free memory

        cls_features = [None] * len(batch_rgbs)
        patch_features = [None] * len(batch_rgbs)
        for index_batch in range(len(batch_rgbs)):
            cls_features[index_batch], patch_features[index_batch] = self.compute_cls_and_patch_features(
                batch_rgbs[index_batch],
                batch_masks[index_batch],
                normalize_cls,
                normalize_patch)  # chunk_size x cF, chunk_size x (P*P) x pF
        cls_features = torch.cat(cls_features, dim=0)       # B x cF
        patch_features = torch.cat(patch_features, dim=0)   # B x (P*P) x pF

        return cls_features, patch_features     # B x cF, B x (P*P) x pF

    @torch.no_grad()
    def forward(self, image_np: np.ndarray, proposals,
                mask_rgbs: bool = True,
                normalize_cls: bool = False, normalize_patch: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param np.ndarray image_np: [H, W, C]
        :param proposals: object with variables: {"boxes": B x 4 (as xyxy), "masks": B x H x W}
        :param mask_rgbs:
        :param bool normalize_cls:
        :param bool normalize_patch:
        :return: A tuple containing the cls tokens [B, cF] and patch tokens [B, P*P, pF]
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        # preprocess image and masks
        processed_rgbs = self.process_rgb_proposals(image_np, proposals.masks if mask_rgbs else None,
                                                    proposals.boxes)                        # B x C x S x S
        processed_masks = self.process_masks_proposals(proposals.masks, proposals.boxes)    # B x S x S

        return self.forward_by_chunk_cls_and_patch_features(processed_rgbs, processed_masks,
                                                            normalize_cls, normalize_patch)     # B x cF, B x (P*P) x pF
