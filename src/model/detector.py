import os
import os.path as osp
import logging
import warnings
import numpy as np
from tqdm import tqdm
import time
import glob
import multiprocessing

import torch
import torchvision.transforms as T
import pytorch_lightning as pl

from src.utils.inout import save_json_bop23
import src.model.scoring as scoring
from src.model.utils import BatchedData, Detections, convert_npz_to_json

from src.model.sam import CustomSamAutomaticMaskGenerator
from src.model.grounded_sam import GroundedSAM

from typing_extensions import Optional, List, Dict, Any, Tuple


class NOCTIS(pl.LightningModule):
    def __init__(self,
                 segmentor_model,
                 descriptor_model,
                 onboarding_config: Dict,
                 matching_config: Dict,
                 post_processing_config: Dict,
                 name_prediction_file: str,
                 log_interval: int,
                 log_dir: str,
                 storage_device: str = torch.device("cpu"),
                 skip_inference: bool = False,
                 save_mask: bool = True,
                 save_score_distribution: bool = False,
                 return_detections: bool = False,
                 reference_dataset: Optional = None,
                 **kwargs):
        super().__init__()

        # sub model for image embedding and mask generation
        self.segmentor_model = segmentor_model
        self.descriptor_model = descriptor_model

        # sub-configurations
        self.onboarding_config = onboarding_config
        self.matching_config = matching_config
        self.post_processing_config = post_processing_config

        # logging
        self.log_interval = log_interval
        self.log_dir = log_dir

        # create log dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.inv_rgb_transform = T.Compose([T.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225])])

        # other
        self.storage_device = storage_device    # use 'torch.device("cuda")' to keep all data in cuda all the time
        self.skip_inference = skip_inference
        self.save_mask = save_mask
        self.save_score_distribution = save_score_distribution
        self.return_detections = return_detections
        self.ref_data = reference_dataset

        self.name_prediction_file = name_prediction_file

        logging.info("Init NOCTIS done!")

    @torch.no_grad()
    def set_reference_objects(self):
        logging.info("Initializing reference objects")

        start_time = time.time()

        descriptors_path = osp.join(self.ref_dataset.template_dir, "descriptors.pth")

        if self.onboarding_config.rendering_type == "pbr":
            # stored in the 'template' dir, but are made from the (some) dataset images.
            descriptors_path = descriptors_path.replace(".pth", "_pbr.pth")

        if os.path.exists(descriptors_path) and not self.onboarding_config.reset_descriptors:
            # load embeddings
            all_ref_features = torch.load(descriptors_path)
            self.ref_data = {"cls_descriptors": all_ref_features["cls_descriptors"].to(self.storage_device),        # O x T x cF
                             "patch_descriptors": all_ref_features["patch_descriptors"].to(self.storage_device)}    # O x T x (P*P) x pF
            logging.info("Loaded descriptors from (cached) file <{}>.".format(descriptors_path))
        else:
            # create descriptors/embeddings of all template images
            self.ref_data = {"cls_descriptors": BatchedData(None),
                             "patch_descriptors": BatchedData(None)}

            for idx in tqdm(range(len(self.ref_dataset)), desc="Computing descriptors ..."):
                ref_data = self.ref_dataset[idx]
                ref_images = ref_data["templates"].to(self.device)         # T x C x S x S
                ref_masks = ref_data["template_masks"].to(self.device)     # T x S x S

                ref_features = self.descriptor_model.compute_cls_and_patch_features(processed_rgbs=ref_images,
                                                                                    masks=ref_masks)

                self.ref_data["cls_descriptors"].append(ref_features[0].to(self.storage_device))    # T x cF
                self.ref_data["patch_descriptors"].append(ref_features[1].to(self.storage_device))  # T x (P*P) x pF

                # free memory and cache clean up
                del ref_images
                del ref_masks
                del ref_features
                torch.cuda.empty_cache()

            self.ref_data["cls_descriptors"].stack()    # O x T x cF
            self.ref_data["cls_descriptors"] = self.ref_data["cls_descriptors"].data
            self.ref_data["patch_descriptors"].stack()  # O x T x (P*P) x pF
            self.ref_data["patch_descriptors"] = self.ref_data["patch_descriptors"].data

            # save the precomputed features for future use
            torch.save(self.ref_data, descriptors_path)

        onboarding_time = time.time() - start_time
        num_objects = len(self.ref_dataset)
        logging.info(("Initializing done. Runtime: {:.02f}s with per object average of: {:.02f},"
                      + " Class descriptors shape: {}, Patch descriptors shape: {}").format(onboarding_time,
                                                                                            onboarding_time/num_objects,
                                                                                            self.ref_data["cls_descriptors"].shape[1:],
                                                                                            self.ref_data["patch_descriptors"].shape[1:]))

    def move_to_device(self):
        self.descriptor_model.model = self.descriptor_model.model.to(self.device)
        self.descriptor_model.model.device = self.device

        if isinstance(self.segmentor_model, CustomSamAutomaticMaskGenerator):
            self.segmentor_model.predictor.model = self.segmentor_model.predictor.model.to(self.device)
        elif isinstance(self.segmentor_model, GroundedSAM):
            self.segmentor_model.move_to_device(self.device)
            # self.segmentor_model.ground_dino.model.to(self.device)
            # self.segmentor_model.sam_predictor.sam.to(self.device)
        else:
            raise ValueError("Unrecognized segmentor model type: <{}>".format(type(self.segmentor_model)))

        logging.info("Moving models to {} done!".format(self.device))

    def aggregate_scores(self,
                         scores: torch.Tensor,
                         aggregation_function: str) -> torch.Tensor:
        """Aggregate the scores using the specified method/function.

        :param torch.Tensor scores:  [B, O, T]
        :param str aggregation_function:
        :return: [B, O]
        :rtype: torch.Tensor
        :raises NotImplementedError: If 'aggregation_function' has an unsupported aggregation function.
        """
        if aggregation_function == "sum":
            # sum of score per object
            # Note: T-dim should sum to 1.0
            score_per_proposal_and_object = torch.sum(scores, dim=-1).clip(0.0, 1.0)    # B x O
        elif aggregation_function == "mean":
            # mean
            score_per_proposal_and_object = torch.mean(scores, dim=-1)                            # B x O
        elif aggregation_function == "median":
            # median
            score_per_proposal_and_object = torch.median(scores, dim=-1)[0]                       # B x O
        elif aggregation_function == "max":
            # max
            score_per_proposal_and_object = torch.max(scores, dim=-1)[0]                          # B x O
        elif "avg_" in aggregation_function:
            # mean over top k scores per object
            k = max(1, min(int(aggregation_function[len("avg_"):]), scores.shape[-1]))
            score_per_proposal_and_object = torch.topk(scores, k=k, dim=-1)[0]                    # B x O x K
            score_per_proposal_and_object = torch.mean(score_per_proposal_and_object, dim=-1)     # B x O
        elif "min_" in aggregation_function:
            # minimum score of the top k scores per object
            k = max(1, min(int(aggregation_function[len("min_"):]), scores.shape[-1]))
            score_per_proposal_and_object = torch.topk(scores, k=k, dim=-1)[0]                    # B x O x K
            score_per_proposal_and_object = torch.min(score_per_proposal_and_object, dim=-1)[0]   # B x O
        elif "knn_" in aggregation_function:
            # knn across objects
            k = float(aggregation_function[len("knn_"):])

            if int(k) != k:
                # relative size
                k = max(0.0, min(k, 1.0))
                k = k * scores.shape[-1]*scores.shape[-2]
            k = max(1, min(int(k), scores.shape[-1]*scores.shape[-2]))

            flat_scores = scores.flatten(start_dim=-2)                                                  # B x (O*T)
            top_scores, top_indices = torch.topk(flat_scores, k=k, dim=-1)                              # B x K, B x K
            top_coords = scoring.convert_flat_index_to_coordinate(top_indices, scores.shape[1:])        # B x K x 2

            score_per_proposal_and_object = torch.zeros(scores.shape[:-1],
                                                        dtype=scores.dtype, device=scores.device)       # B x O
            score_per_proposal_and_object = score_per_proposal_and_object.scatter_add(dim=1,
                                                                                      index=top_coords[:, :, 0],
                                                                                      src=top_scores)    # B x O
            score_per_proposal_and_object /= top_scores.sum(dim=-1, keepdim=True)
        else:
            raise NotImplementedError("Unknown requested aggregation function of name: <{}>".format(aggregation_function))

        return score_per_proposal_and_object    # B x O

    def choose_best_and_filter(self,
                               score_prop_obj: torch.Tensor,
                               min_score_threshold: float,
                               max_num_instances: Optional[int] = 100) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param torch.Tensor score_prop_obj:  [B, O]
        :param float min_score_threshold:
        :param Optional[int] max_num_instances:
        :return A tuple containing indices of selected proposals, there assigned object ids, there predicted score/confidence and a confidence matrix [B, O]
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        # assign each proposal to the object with the highest scores
        score_per_proposal, assigned_object_indices = torch.max(score_prop_obj, dim=-1)     # B, B

        # keep only proposals with large enough confidence threshold
        selected_proposal_indices = torch.arange(
            len(score_per_proposal),
            device=score_per_proposal.device)[score_per_proposal > min_score_threshold]  # B'

        # for bop challenge, we only keep top 'max_num_instances' (normally 100) instances
        if max_num_instances is not None and len(selected_proposal_indices) > max_num_instances:
            logging.info("Select only top {} instances for detection.".format(max_num_instances))
            _, indices = torch.topk(score_per_proposal[selected_proposal_indices],
                                    k=max_num_instances)                    # max_num_instances
            selected_proposal_indices = selected_proposal_indices[indices]  # max_num_instances

        pred_object_indices = assigned_object_indices[selected_proposal_indices]    # B'
        pred_scores = score_per_proposal[selected_proposal_indices]                 # B'
        pred_score_distribution = score_prop_obj[selected_proposal_indices]         # B' x O

        return selected_proposal_indices, pred_object_indices, pred_scores, pred_score_distribution  # B', B', B', B' x O

    def compute_semantic_score(self, query_cls_descriptors: torch.Tensor) -> torch.Tensor:
        """
        :param query_cls_descriptors: [B, cF]
        :return: sematic scores [B, O]
        :rtype: torch.Tensor
        """
        # compute semantic matching scores for each proposal
        scores = scoring.compute_semantic_similarity(query_cls_descriptors,
                                                     self.ref_data["cls_descriptors"],
                                                     self.matching_config.semantic_confidence_threshold)    # B x O x T

        #scores[scores <= 0.2] = 0.0
        score_per_proposal_and_object = self.aggregate_scores(scores,
                                                              self.matching_config.semantic_aggregation_function)  # B x O

        return score_per_proposal_and_object    # B x O

    def compute_appearance_score(self, query_patch_descriptors: torch.Tensor) -> torch.Tensor:
        """
        :param query_patch_descriptors: [B, (P*P), pF]
        :return: appearance scores [B, O]
        :rtype: torch.Tensor
        """
        # compute appearance matching scores for each proposal
        scores = scoring.compute_appearance_similarity(query_patch_descriptors,
                                                       self.ref_data["patch_descriptors"],
                                                       self.matching_config.appearance_confidence_threshold,
                                                       self.matching_config.appearance_cycle_threshold)  # B x O x T

        if self.matching_config.appearance_score_extra_weight_factor != 1.0:
            # give extra weight
            scores *= self.matching_config.appearance_score_extra_weight_factor
            #scores.clamp_(min=0.0, max=1.0)

        score_per_proposal_and_object = self.aggregate_scores(scores,
                                                              self.matching_config.appearance_aggregation_function)  # B x O

        return score_per_proposal_and_object    # B x O

    def test_step(self, batch, idx: int):
        if idx == 0:
            self.move_to_device()
            self.set_reference_objects()
        assert batch["image"].shape[0] == 1, "Batch size must be 1"

        if self.skip_inference:
            return

        # inverse transform back to normal image
        image_np = self.inv_rgb_transform(batch["image"][0]).detach().cpu().numpy().transpose(1, 2, 0)   # H x W x C
        image_np = np.uint8(image_np.clip(0, 1) * 255)

        # create proposals
        proposal_stage_start_time = time.time()
        proposals = self.segmentor_model.generate_masks(image_np)   # {'boxes': B x 4 (as xyxy), 'masks': B x H x W}

        torch.cuda.empty_cache()  # clean up cache

        logging.info("Made {} detections".format(len(proposals["boxes"])))

        # init detections with masks and boxes
        detections = Detections(proposals)
        detections.remove_very_small_detections(config=self.post_processing_config.mask_post_processing)

        # compute cls descriptors and patch descriptors for every query proposals
        query_cls_descriptors, query_patch_descriptors = self.descriptor_model(image_np, detections)    # B x cF, B x (P*P) x pF
        proposal_stage_end_time = time.time()

        torch.cuda.empty_cache()  # clean up cache

        # matching descriptors
        matching_stage_start_time = time.time()

        semantic_score_weight_factor = max(0, self.matching_config.semantic_score_weight_factor)
        appearance_score_weight_factor = max(0, self.matching_config.appearance_score_weight_factor)

        # compute the sematic score
        if semantic_score_weight_factor > 0:
            sem_score = self.compute_semantic_score(query_cls_descriptors)      # B x O
        else:
            sem_score = 0

        # compute the appearance score
        if appearance_score_weight_factor > 0:
            app_score = self.compute_appearance_score(query_patch_descriptors)      # B x O
        else:
            app_score = 0

        # final score
        final_score = (semantic_score_weight_factor*sem_score
                       + appearance_score_weight_factor*app_score)      # B x O
        final_score /= (semantic_score_weight_factor
                        + appearance_score_weight_factor)               # B x O

        if self.matching_config.use_detector_confidence:
            # include the confidence of queries/proposals
            conf = torch.zeros(final_score.shape[0],
                               dtype=final_score.dtype,
                               device=final_score.device)   # B
            num_conf_factors = 0
            if hasattr(detections, "boxes_scores"):
                conf += detections.boxes_scores             # B
                num_conf_factors += 1
            if hasattr(detections, "masks_scores"):
                conf += detections.masks_scores             # B
                num_conf_factors += 1
            conf /= num_conf_factors                        # B

            final_score *= conf[:, None]                    # B x O

        final_score.clamp_(min=0.0, max=1.0)

        # choose best match and update detections
        (
            selected_proposal_indices,  # B'
            pred_object_indices,        # B'
            pred_scores,                # B'
            pred_score_distribution     # B' x O
        ) = self.choose_best_and_filter(final_score,
                                        self.matching_config.final_confidence_threshold,
                                        self.matching_config.max_num_instances)

        detections.filter(selected_proposal_indices)
        detections.add_attribute("scores", pred_scores)
        detections.add_attribute("score_distribution", pred_score_distribution)
        detections.add_attribute("object_ids", pred_object_indices)
        detections.apply_nms_per_object_id(self.post_processing_config.nms_threshold)
        matching_stage_end_time = time.time()

        runtime = (proposal_stage_end_time - proposal_stage_start_time
                   + matching_stage_end_time - matching_stage_start_time)
        detections.to_numpy()

        # map object indices to object ids
        detections.object_ids = np.array(self.ref_dataset.obj_ids)[detections.object_ids]

        logging.info("{} objects where identified.".format(len(detections)))

        if self.return_detections:
            return {"detections": detections,
                    "runtime_proposal_stage": proposal_stage_end_time - proposal_stage_start_time,
                    "runtime_matching_stage": matching_stage_end_time - matching_stage_start_time}
        else:
            # store detection as 'npz.' file
            scene_id = batch["scene_id"][0]
            frame_id = batch["frame_id"][0]

            # save detections to file
            save_dir = osp.join(self.log_dir, "predictions", self.dataset_name, self.name_prediction_file)
            os.makedirs(save_dir, exist_ok=True)

            file_path = osp.join(save_dir, "scene_{}_frame_{}".format(scene_id, frame_id))
            results = detections.save_to_file(scene_id=int(scene_id),
                                              frame_id=int(frame_id),
                                              runtime=runtime,
                                              file_path=file_path,
                                              return_results=True,
                                              save_mask=self.save_mask,
                                              save_score_distribution=self.save_score_distribution)
            # save runtime to file
            np.savez(file_path + "_runtime",
                     proposal_stage=proposal_stage_end_time - proposal_stage_start_time,
                     matching_stage=matching_stage_end_time - matching_stage_start_time)
            return 0

    def on_test_epoch_end(self):
        if not self.return_detections and self.global_rank == 0:
            # only rank 0 process
            # load the results from all files
            result_paths = sorted(glob.glob(osp.join(self.log_dir, "predictions",
                                                     self.dataset_name, self.name_prediction_file, "*.npz")))
            result_paths = sorted([path for path in result_paths if "runtime" not in path])

            num_workers = 10
            logging.info("Combine 'npz'-files to final result using {} workers.".format(num_workers))

            formatted_detections = []
            formatted_detections_with_score_distribution = []

            if num_workers > 0:
                # with multiprocessing
                pool = multiprocessing.Pool(processes=num_workers)
                for detection in tqdm(pool.imap_unordered(convert_npz_to_json, result_paths),
                                      total=len(result_paths),
                                      desc="Load and converting 'npz' to 'json'"):
                    formatted_detections.extend(detection[0])
                    formatted_detections_with_score_distribution.extend(detection[1])
            else:
                # without multiprocessing
                for detection in tqdm(map(convert_npz_to_json, result_paths),
                                      total=len(result_paths),
                                      desc="Load and converting 'npz' to 'json'"):
                    formatted_detections.extend(detection[0])
                    formatted_detections_with_score_distribution.extend(detection[1])

            detections_path = osp.join(self.log_dir, "{}.json".format(self.name_prediction_file))
            os.makedirs(osp.dirname(detections_path), exist_ok=True)
            save_json_bop23(detections_path, formatted_detections)
            logging.info("Saved final predictions to <{}>.".format(detections_path))

            if self.save_score_distribution:
                detections_path = osp.join(self.log_dir, "{}_with_score_distribution.json".format(self.name_prediction_file))
                save_json_bop23(detections_path, formatted_detections_with_score_distribution)
                logging.info("Saved final predictions to <{}>.".format(detections_path))
