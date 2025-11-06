import os
import os.path as osp
import argparse

from collections import defaultdict

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import distinctipy

from src.utils.inout import load_json
from src.utils.visualization_detectron2 import NOCTISVisualizer
from src.utils.visualization_utils import visualize_object
from segment_anything.utils.amg import rle_to_mask

from typing import List, Optional

if True:
    # set random number generator seed
    import random
    import numpy
    import torch
    random.seed(2025)
    numpy.random.seed(2025)
    torch.manual_seed(2025)


def visualize(input_json_file: str,
              dataset_dir: str, dataset_split: str,
              rgb_name: str, rgb_prefix: str, rgb_suffix: str,
              ignore_object_ids: List[int],
              minimum_scores: List[float],
              easy_mode: bool,
              object_names: List[str], with_labels: bool,
              jitter_colors: bool,
              background: str, split_image: bool,
              output_dir: Optional[str],
              output_scene_dir_prefix: str, output_scene_dir_suffix: str,
              output_image_prefix: str, output_image_suffix: str) -> None:
    print("Start loading detections.")

    # load detection results from .json
    # [{"scene_id": int, "image_id": int, "category_id": int, "bbox": [4*int xywh], "score": float, "time": float,
    #   "segmentation": {"counts": [N*int], "size": [2*int]}}]
    dets = load_json(input_json_file)

    print("Loaded {} detections.".format(len(dets)))

    # load all possible object ids
    for first_dir in ["models", ""]:
        for second_dir in ["models", "models_cad", "object_models"]:
            models_info_file = osp.join(dataset_dir, first_dir, second_dir, "models_info.json")
            if osp.exists(models_info_file):
                break
        else:
            continue
        break
    else:
        raise ValueError("'models_info.json' could not be found.")

    if ignore_object_ids is None:
        ignore_object_ids = []

    index_to_obj_id = sorted({int(obj_id) for obj_id in load_json(models_info_file).keys()})
    index_to_obj_id = [i for i in index_to_obj_id if i not in ignore_object_ids]    # filter out unwanted object ids
    obj_id_to_index = {obj_id: i for (i, obj_id) in enumerate(index_to_obj_id)}

    # filter out all detection, whose confidence score is too low, use first 'minimum_scores' as default
    obj_id_minimum_scores = defaultdict(lambda: minimum_scores[0], zip(index_to_obj_id, minimum_scores))
    filtered_dets = [det for det in dets if det["score"] > obj_id_minimum_scores[det["category_id"]]]

    # use 'object_<id>' as default object name
    if object_names is None:
        object_names = []

    obj_id_obj_names = ["object_{}".format(i) for i in index_to_obj_id]     # object_{:06d} is too long as name
    n = min(len(object_names), len(index_to_obj_id))
    obj_id_obj_names[:n] = object_names[:n]

    obj_id_obj_colors = distinctipy.get_colors(len(index_to_obj_id))

    print("Keeping only {} detections after minimum score filtering.".format(len(filtered_dets)))

    # sort detections by 'scene_id' and 'image_id'
    list_scene_id_and_image_id = [(det["scene_id"], det["image_id"]) for det in filtered_dets]
    list_scene_id_and_image_id = set(list_scene_id_and_image_id)
    list_scene_id_and_image_id = sorted(list_scene_id_and_image_id)

    print("Start creating annotated images.")

    dataset_name = os.path.basename(dataset_dir)

    for (scene_id, image_id) in tqdm(list_scene_id_and_image_id):
        image_file = "{}/{}/{:06d}/{}/{}{:06d}.{}".format(dataset_dir, dataset_split, scene_id,
                                                          rgb_name, rgb_prefix, image_id, rgb_suffix)
        image = np.array(Image.open(image_file).convert("RGB"))     # H x W x 3, convert gray images to rgb to prevent color distortion

        if split_image:
            rgb = np.copy(image)                        # H x W x 3

        if background == "rgb":
            pass
        elif background == "gray":
            # convert to gray image
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)     # H x W x 3
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)     # H x W x 3
        elif background == "black":
            image = np.zeros_like(image, dtype=np.uint8)        # H x W x 3
        elif background == "white":
            image = np.ones_like(image, dtype=np.uint8)         # H x W x 3
        else:
            raise NotImplementedError("Value for 'background' must be one of ['rgb', 'gray', 'black', 'white'], but saw <{}>".format(background))

        # collect all detection data of the current image
        object_ids = []
        masks = []
        scores = []
        bboxes = []
        for det in filtered_dets:
            if scene_id == det["scene_id"] and image_id == det["image_id"]:
                object_ids.append(obj_id_to_index[det["category_id"]])
                masks.append(rle_to_mask(det["segmentation"]))      # H x W
                scores.append(det["score"])
                bbox = det["bbox"]      # xywh
                bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]     # xyxy
                bboxes.append(bbox)

        object_ids = np.array(object_ids)   # B
        masks = np.stack(masks)             # B x H x W
        scores = np.array(scores)           # B
        bboxes = np.array(bboxes)           # B

        if easy_mode:
            # draw object mask and contour only
            vis = visualize_object(img=image,
                                   masks=masks,
                                   labels=object_ids,
                                   colors=obj_id_obj_colors,
                                   jitter_colors=jitter_colors) # H x W x 3
        else:
            # use detectron2 for visualization
            # draw bounding boxes and object information
            visualizer = NOCTISVisualizer(obj_names=obj_id_obj_names,
                                          obj_colors=obj_id_obj_colors,
                                          img_size=image.shape[:2],
                                          jitter_colors=jitter_colors,
                                          with_labels=with_labels)

            vis = visualizer.forward(rgb=image,
                                     masks=masks,
                                     bboxes=bboxes,
                                     scores=scores,
                                     labels=object_ids,
                                     save_path=None)      # H x W x 3

        if split_image:
            # combine side-by-side original image with annotated version
            vis = np.concatenate([rgb, vis], axis=1)    # H x (2*W) x 3

        output_image_scene_dir = "{}/{}{}{:06d}{}".format(output_dir, output_scene_dir_prefix,
                                                          dataset_name, scene_id, output_scene_dir_suffix)
        os.makedirs(output_image_scene_dir, exist_ok=True)
        output_image_file = "{}/{}{:06d}.{}".format(output_image_scene_dir, output_image_prefix,
                                                    image_id, output_image_suffix)

        vis = Image.fromarray(vis)
        vis.save(output_image_file)

    print("Finished creating the annotated images.")
    print("End")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creates a simple 'scene_camera.json' for the current scene.")

    parser.add_argument("-i", "--input_json_file", dest="input_json_file", type=str, required=True,
                        help="Path of the result '.json' file containing the detections.")

    parser.add_argument("-d", "--dataset_dir", dest="dataset_dir", type=str, default=".",
                        help="Path to dataset to use.")
    parser.add_argument("-s", "--dataset_split", dest="dataset_split", type=str, default="test",
                        help="Dataset split ['train', 'val', 'test', ...] to use.")

    parser.add_argument("-u", "--rgb_name", dest="rgb_name", type=str, default="rgb",
                        help="Dirname, where rgb image are stored.")
    parser.add_argument("-v", "--rgb_prefix", dest="rgb_prefix", type=str, default="",
                        help="Prefix of the rgb image before the actual id.")
    parser.add_argument("-w", "--rgb_suffix", dest="rgb_suffix", type=str, default="png",
                        help="Suffix of the rgb image.")

    parser.add_argument("-c", "--ignore_object_ids", dest="ignore_object_ids", type=int, nargs="+",
                        help="List of object ids/classes to ignore.")

    parser.add_argument("-m", "--minimum_scores", dest="minimum_scores", type=float, nargs="+", default=[0.5],
                        help="The minimum confidence score for each object id/class to be shown. First score is used as default for missing classes")

    parser.add_argument("-e", "--easy_mode", dest="easy_mode", action="store_true", default=False,
                        help="Does not use detectron2 to visualize the annotations.")

    parser.add_argument("-n", "--object_names", dest="object_names", type=str, nargs="+",
                        help="List of object id/class to object name.")

    parser.add_argument("-l", "--without_labels", dest="without_labels", action="store_true", default=False,
                        help="Annotates the objects without an label (no name and score).")

    parser.add_argument("-j", "--jitter_colors", dest="jitter_colors", action="store_true", default=False,
                        help="Jitters the color of each object class to make multiple occurrences more different.")
    parser.add_argument("-b", "--background", dest="background", choices=["rgb", "gray", "black", "white"], default="gray",
                        help="Choose which background image for the renderings to use: 'rgb': scene image, 'gray': grayscale scene image, 'black'/'white': black/white background.")
    parser.add_argument("-g", "--split_image", dest="split_image", action="store_true", default=False,
                        help="Result image is the original image side by side with rendered overlay image.")

    parser.add_argument("-o", "--output_dir", dest="output_dir", type=str, default=None,
                        help="Path of the output dir to save the rendered images in.")

    parser.add_argument("-p", "--output_scene_dir_prefix", dest="output_scene_dir_prefix", type=str, default="",
                        help="Prefix of the created scene output dir.")
    parser.add_argument("-q", "--output_scene_dir_suffix", dest="output_scene_dir_suffix", type=str, default="",
                        help="Suffix of the created scene output dir.")

    parser.add_argument("-r", "--output_image_prefix", dest="output_image_prefix", type=str, default="",
                        help="Prefix of the created output images.")
    parser.add_argument("-t", "--output_image_suffix", dest="output_image_suffix", type=str, default="png",
                        help="Suffix of the created output images.")

    args = parser.parse_args()

    visualize(input_json_file=args.input_json_file,
              dataset_dir=args.dataset_dir,
              dataset_split=args.dataset_split,
              rgb_name=args.rgb_name,
              rgb_prefix=args.rgb_prefix,
              rgb_suffix=args.rgb_suffix,
              ignore_object_ids=args.ignore_object_ids,
              minimum_scores=args.minimum_scores,
              easy_mode=args.easy_mode,
              object_names=args.object_names,
              with_labels=not args.without_labels,
              jitter_colors=args.jitter_colors,
              background=args.background,
              split_image=args.split_image,
              output_dir=args.output_dir,
              output_scene_dir_prefix=args.output_scene_dir_prefix,
              output_scene_dir_suffix=args.output_scene_dir_suffix,
              output_image_prefix=args.output_image_prefix,
              output_image_suffix=args.output_image_suffix)

