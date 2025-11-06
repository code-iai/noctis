import logging
import numpy as np

import torch
import torchvision
from torchvision.ops.boxes import box_area

from src.utils.inout import save_npz
from src.utils.bbox_utils import xyxy_to_xywh, xywh_to_xyxy, force_binary_mask

from typing_extensions import Any, Union, Dict, List, Optional


def mask_to_rle(binary_mask: np.ndarray) -> Dict:
    """Converts a binary image mask into the (COCO) RLE format.

    Note: Based on "https://stackoverflow.com/a/76990451"

    :param binary_mask: H x W  (also H x W x 1 or 1 x H x W)
    """
    rle = {"counts": [], "size": list(binary_mask.squeeze().shape)}

    flattened_mask = binary_mask.ravel(order="F")
    diff_arr = np.diff(flattened_mask)
    nonzero_indices = np.where(diff_arr != 0)[0] + 1
    lengths = np.diff(np.concatenate(([0], nonzero_indices, [len(flattened_mask)])))

    # note that the odd counts are always the numbers of zeros
    if flattened_mask[0] == 1:
        lengths = np.concatenate(([0], lengths))

    rle["counts"] = lengths.tolist()

    return rle      # {'size': 2, 'counts': N}


class BatchedData:
    """
    A structure for storing data in batched format.
    Implements basic functionality for appending and final concatenation.
    """

    def __init__(self, batch_size: int, data: Optional[List] = None) -> None:
        self.batch_size = batch_size

        if data is not None:
            self.data = data
        else:
            self.data = []

    def __len__(self) -> int:
        assert self.batch_size is not None, "batch_size is not defined"
        return np.ceil(len(self.data) / self.batch_size).astype(int)

    def __getitem__(self, idx) -> Any:
        assert self.batch_size is not None, "batch_size is not defined"
        return self.data[idx * self.batch_size:(idx + 1) * self.batch_size]

    def cat(self, data: Any, dim: int = 0) -> None:
        if len(self.data) == 0:
            self.data = data
        else:
            self.data = torch.cat([self.data, data], dim=dim)

    def append(self, data: Any) -> None:
        self.data.append(data)

    def stack(self, dim: int = 0) -> None:
        self.data = torch.stack(self.data, dim=dim)


class Detections:
    """
    A structure for storing detections.
    """

    def __init__(self, data: Union[str, Dict]) -> None:
        if isinstance(data, str):
            # load data from file
            data = Detections.load_from_file(data)

        for key, value in data.items():
            setattr(self, key, value)

        self.keys = list(data.keys())

        if "boxes" in self.keys:
            if isinstance(self.boxes, np.ndarray):
                self.to_torch()
            self.boxes = self.boxes.long()

    def remove_very_small_detections(self, config: Dict) -> None:
        img_area = self.masks.shape[1] * self.masks.shape[2]
        box_areas = box_area(self.boxes) / img_area
        mask_areas = self.masks.sum(dim=(1, 2)) / img_area
        keep_indices = torch.logical_and(
            #box_areas > config.min_box_size**2, mask_areas > config.min_mask_size
            box_areas > config.min_box_size**2, mask_areas.to(box_areas.device) > config.min_mask_size
        )

        logging.info("Removing {} to small detections.".format(len(keep_indices) - keep_indices.sum()))
        self.filter(keep_indices)

    def apply_nms_per_object_id(self, nms_threshold: float = 0.5) -> None:
        """
        Performs non-maximum suppression (NMS) on the bounding boxes per object according to their score.
        """
        keep_indices = []
        all_indexes = torch.arange(len(self.object_ids), device=self.boxes.device)

        for object_id in torch.unique(self.object_ids):
            index = self.object_ids == object_id
            object_id_indices = all_indexes[index]
            keep_index = torchvision.ops.nms(self.boxes[index].float(), self.scores[index].float(), nms_threshold)
            keep_indices.append(object_id_indices[keep_index])

        if len(keep_indices):
            keep_indices = torch.cat(keep_indices)

            logging.info("Removing {} minimum score/overlapping detections.".format(len(self.object_ids) - len(keep_indices)))
            self.filter(keep_indices)

    def apply_nms(self, nms_threshold: float = 0.5) -> None:
        """
        Performs non-maximum suppression on all bounding boxes according to their score
        """
        keep_indices = torchvision.ops.nms(self.boxes.float(), self.scores.float(), nms_threshold)
        self.filter(keep_indices)

    def add_attribute(self, key: str, value: Any) -> None:
        setattr(self, key, value)
        self.keys.append(key)

    def __len__(self) -> int:
        return len(self.boxes)

    def check_size(self):
        mask_size = len(self.masks)
        box_size = len(self.boxes)
        score_size = len(self.scores)
        object_id_size = len(self.object_ids)

        assert (
            mask_size == box_size == score_size == object_id_size
        ), "Size mismatch {} {} {} {}".format(mask_size, box_size, score_size, object_id_size)

    def to_numpy(self) -> None:
        for key in self.keys:
            setattr(self, key, getattr(self, key).detach().cpu().numpy())

    def to_torch(self) -> None:
        for key in self.keys:
            setattr(self, key, torch.from_numpy(getattr(self, key)))

    def save_to_file(self, scene_id: int, frame_id: int, runtime: float, file_path: str,
                     return_results: bool = False, save_mask: bool = True,
                     save_score_distribution: bool = False) -> Optional[Dict]:
        """
        scene_id, image_id, category_id, bbox, time
        """
        boxes = xyxy_to_xywh(self.boxes)
        results = {"scene_id": scene_id,
                   "image_id": frame_id,
                   "category_id": self.object_ids,
                   "score": self.scores,
                   "bbox": boxes,
                   "time": runtime}

        if save_mask:
            results["segmentation"] = self.masks
        if save_score_distribution:
            assert hasattr(self, "score_distribution"), "score_distribution is not defined"
            results["score_distribution"] = self.score_distribution

        save_npz(file_path, results)

        if return_results:
            return results

    @staticmethod
    def load_from_file(file_path: str) -> Dict:
        data = np.load(file_path)

        # extract data
        masks = data["segmentation"]
        boxes = xywh_to_xyxy(np.array(data["bbox"]))
        data = {"object_ids": data["category_id"] - 1,
                "bbox": boxes,
                "scores": data["score"],
                "masks": masks}

        logging.info("Loaded {}".format(file_path))

        return data

    def filter(self, indices: Union[np.ndarray, torch.Tensor]):
        for key in self.keys:
            #setattr(self, key, getattr(self, key)[idxs])
            value = getattr(self, key)
            setattr(self, key, value[torch.as_tensor(indices, device=value.device)])

    def clone(self):
        """
        Clone the current object
        """
        return Detections(self.__dict__.copy())


def convert_npz_to_json(npz_path: str) -> Dict:
    detections = np.load(npz_path)

    results = []
    results_with_score_distribution = []

    for idx_det in range(len(detections["bbox"])):
        result = {"scene_id": int(detections["scene_id"]),
                  "image_id": int(detections["image_id"]),
                  "category_id": int(detections["category_id"][idx_det]),
                  "bbox": detections["bbox"][idx_det].tolist(),
                  "score": float(detections["score"][idx_det]),
                  "time": float(detections["time"])}

        if "segmentation" in detections.keys():
            result["segmentation"] = mask_to_rle(force_binary_mask(detections["segmentation"][idx_det]))

        results.append(result)

        if "score_distribution" in detections.keys():
            result_with_score_distribution = result.copy()
            result_with_score_distribution["score_distribution"] = detections["score_distribution"][idx_det].tolist()
            results_with_score_distribution.append(result_with_score_distribution)

    return results, results_with_score_distribution
