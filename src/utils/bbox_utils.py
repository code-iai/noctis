import numpy as np

import torch
import torch.nn.functional as F

from typing_extensions import Tuple, Union


class CropResizePad:
    def __init__(self, target_size: Union[int, Tuple[int, int]], pad_value: float = 0.0) -> None:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self.target_size = target_size
        self.target_ratio = self.target_size[1] / self.target_size[0]
        self.target_h, self.target_w = target_size
        self.target_max = max(self.target_h, self.target_w)
        self.pad_value = pad_value

    def __call__(self, images: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """
        :param torch.Tensor images: [B, C, H, W]
        :param torch.Tensor boxes: [B, 4] as xyxy
        :return: Cropped and padded images
        :rtype: torch.Tensor
        """
        box_sizes = boxes[:, 2:4] - boxes[:, 0:2]                           # B x 2
        scale_factor = torch.minimum(self.target_h / box_sizes[:, 0],
                                     self.target_w / box_sizes[:, 1])       # B

        processed_images = [None] * images.shape[0]

        for i, (image, box, scale) in enumerate(zip(images, boxes, scale_factor)):
            # crop and scale
            image = image[:, box[1]:box[3], box[0]:box[2]]      # C x bH x bW

            # don't scale image dimensions equal 1, if scale <= 0, because they would be 0 after rounding down
            scale = scale.item()
            height_scale = 1.0 if scale < 1.0 and image.shape[1] == 1 else scale
            width_scale = 1.0 if scale < 1.0 and image.shape[2] == 1 else scale

            # interpolate
            image = F.interpolate(image.unsqueeze(0), scale_factor=(height_scale, width_scale))[0]  # C x iH x iW

            # pad and resize
            original_h, original_w = image.shape[1:]
            original_ratio = original_w / original_h

            # check if the original and final aspect ratios are the same within a margin
            if self.target_ratio != original_ratio:
                padding_top = max((self.target_h - original_h) // 2, 0)
                padding_bottom = self.target_h - original_h - padding_top
                padding_left = max((self.target_w - original_w) // 2, 0)
                padding_right = self.target_w - original_w - padding_left

                image = F.pad(image, (padding_left, padding_right, padding_top, padding_bottom),
                              value=self.pad_value)     # C x tH x tW

            # sometimes one pixel can be lost due to rounding
            image = F.interpolate(image.unsqueeze(0), size=(self.target_h, self.target_w))[0]   # C x tH x tW

            processed_images[i] = image

        return torch.stack(processed_images, dim=0)     # B x C x tH x tW


def xyxy_to_xywh(bbox):
    """Convert [x1, y1, x2, y2] box format to [x, y, w, h] format."""
    if len(bbox.shape) == 1:
        x1, y1, x2, y2 = bbox
        return [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
    elif len(bbox.shape) == 2:
        x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        return np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
    else:
        raise ValueError("bbox must be a numpy array of shape (4,) or (N, 4), but saw <{}>.".format(bbox.shape))


def xywh_to_xyxy(bbox):
    """Convert [x, y, w, h] box format to [x1, y1, x2, y2] format."""
    if len(bbox.shape) == 1:
        x, y, w, h = bbox
        return [x, y, x + w - 1, y + h - 1]
    elif len(bbox.shape) == 2:
        x, y, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        return np.stack([x, y, x + w, y + h], axis=1)
    else:
        raise ValueError("bbox must be a numpy array of shape (4,) or (N, 4), but saw <{}>.".format(bbox.shape))


def get_bbox_size(bbox):
    return [bbox[2] - bbox[0], bbox[3] - bbox[1]]


def make_bbox_dividable(bbox_size, dividable_size, ceil=True):
    if ceil:
        new_size = np.ceil(np.array(bbox_size) / dividable_size) * dividable_size
    else:
        new_size = np.floor(np.array(bbox_size) / dividable_size) * dividable_size
    return new_size


def make_bbox_square(old_bbox):
    size_to_fit = np.max([old_bbox[2] - old_bbox[0], old_bbox[3] - old_bbox[1]])
    new_bbox = np.array(old_bbox)
    old_bbox_size = [old_bbox[2] - old_bbox[0], old_bbox[3] - old_bbox[1]]

    # Add padding into y axis
    displacement = int((size_to_fit - old_bbox_size[1]) / 2)
    new_bbox[1] = old_bbox[1] - displacement
    new_bbox[3] = old_bbox[3] + displacement

    # Add padding into x axis
    displacement = int((size_to_fit - old_bbox_size[0]) / 2)
    new_bbox[0] = old_bbox[0] - displacement
    new_bbox[2] = old_bbox[2] + displacement

    return new_bbox


def crop_image(image, bbox, format="xyxy"):
    if format == "xyxy":
        image_cropped = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
    elif format == "xywh":
        image_cropped = image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :]
    return image_cropped


def force_binary_mask(mask, threshold=0.):
    mask = np.where(mask > threshold, 1, 0)
    return mask


def get_bbox_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def compute_iou(bb_a, bb_b):
    """Calculates the Intersection over Union (IoU) of two 2D bounding boxes.

    :param bb_a: 2D bounding box (x1, y1, x2, y2)
    :param bb_b: 2D bounding box (x2, y2, x2, y2)
    :return: The IoU value.
    """
    # intersection rectangle
    tl_inter = torch.max(bb_a[:, 0:2], bb_b[:, 0:2])
    br_inter = torch.min(bb_a[:, 2:4], bb_b[:, 2:4])
    wh_a = bb_a[:, 2:4] - bb_a[:, 0:2]
    wh_b = bb_b[:, 2:4] - bb_b[:, 0:2]

    # width and height of the intersection rectangle
    wh_inter = br_inter - tl_inter

    if (wh_inter > 0).all():
        area_inter = wh_inter[:, 0] * wh_inter[:, 1]
        area_a = wh_a[:, 0] * wh_a[:, 1]
        area_b = wh_b[:, 0] * wh_b[:, 1]
        iou = area_inter / (area_a + area_b - area_inter)
    else:
        iou = 0.0

    return iou


def compute_bbox_iou(bbox_a: torch.Tensor, bbox_b: torch.Tensor) -> torch.Tensor:
    """Calculates the Intersection over Union (IoU) of two 2D bounding boxes.

    :param torch.Tensor bbox_a: 2D bounding boxes [B, 4] as xyxy
    :param torch.Tensor bbox_b: 2D bounding boxes [B, 4] as xyxy
    :return [B]
    :rtype: torch.Tensor
    """
    # intersection area
    tl_corner_inter = torch.maximum(bbox_a[:, 0:2], bbox_b[:, 0:2])     # B x 2
    br_corner_inter = torch.minimum(bbox_a[:, 2:4], bbox_b[:, 2:4])     # B x 2
    wh_inter = br_corner_inter - tl_corner_inter  # B x 2
    area_inter = wh_inter[:, 0] * wh_inter[:, 1]

    # area of bounding box A and B
    wh_a = bbox_a[:, 2:4] - bbox_a[:, 0:2]  # B x 2
    area_a = wh_a[:, 0] * wh_a[:, 1]        # B x 2

    wh_b = bbox_b[:, 2:4] - bbox_b[:, 0:2]  # B x 2
    area_b = wh_b[:, 0] * wh_b[:, 1]        # B x 2

    # intersection over union
    iou = area_inter / (area_a + area_b - area_inter)   # B

    # no intersection area results in an iou of 0
    iou[area_inter <= 0] = 0

    return iou  # B
