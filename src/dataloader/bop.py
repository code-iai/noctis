import logging
import os
import os.path as osp
from tqdm import tqdm
import time
from pathlib import Path
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import pandas as pd

from src.poses.utils import load_indices_level_in_level_mapping
from src.utils.bbox_utils import CropResizePad
from src.utils.inout import load_json, save_json, casting_format_to_save_json


from typing_extensions import Dict, List, Optional, Union, Tuple


class BaseBOP(Dataset):
    def __init__(self, root_dir: str, split: str) -> None:
        """
        Read a dataset in the BOP format.
        See https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md
        """
        self.root_dir = root_dir
        self.split = split

        self.list_scenes = None
        self.metadata = None

    def load_list_scene(self, split: Optional[Union[str, List[str]]] = None) -> None:
        if isinstance(split, str):
            if split is not None:
                split_folder = osp.join(self.root_dir, split)
            self.list_scenes = sorted([osp.join(split_folder, scene) for scene in os.listdir(split_folder)
                                       if osp.isdir(osp.join(split_folder, scene)) and scene != "models"])
        elif isinstance(split, list):
            self.list_scenes = []
            for scene in split:
                if not isinstance(scene, str):
                    scene = "{:06d}".format(scene)
                if os.path.isdir(osp.join(self.root_dir, scene)):
                    self.list_scenes.append(osp.join(self.root_dir, scene))
            self.list_scenes = sorted(self.list_scenes)
        else:
            raise NotImplementedError

        logging.info("Found {} scenes".format(len(self.list_scenes)))

    def load_metadata(self, reset_metadata: bool = True, split: str = "test", load_gt: bool = False,
                      shuffle_data: bool = False) -> None:
        start_time = time.time()

        logging.info("Loading metadata for split {}".format(split))

        metadata_path = osp.join(self.root_dir, "{}_metadata.json".format(split))

        if reset_metadata or not osp.exists(metadata_path):
            json_names = ["scene_camera"]
            mask_types = []
            if load_gt:
                json_names.extend(["scene_gt", "scene_gt_info"])
                mask_types = ["mask", "mask_visib"]

            metadata = {v: [] for v in ["scene_id", "frame_id", "rgb_path", "depth_path"] + json_names + mask_types}

            for scene_path in tqdm(self.list_scenes, desc="Loading metadata"):
                scene_id = scene_path.split("/")[-1]

                if osp.exists(osp.join(scene_path, "rgb")):
                    rgb_paths = sorted(Path(scene_path).glob("rgb/*.[pj][pn][g]"))
                    depth_paths = sorted(Path(scene_path).glob("depth/*.[pj][pn][g]"))
                else:
                    rgb_paths = sorted(Path(scene_path).glob("gray/*.tif"))
                    depth_paths = sorted(Path(scene_path).glob("depth/*.tif"))

                rgb_paths = [str(x) for x in rgb_paths]
                depth_paths = [str(x) for x in depth_paths]

                if len(depth_paths):
                    depth_suffix = osp.splitext(depth_paths[0])[1][1:]  # take suffix from one depth image as representative
                else:
                    # default
                    depth_suffix = "png"

                assert len(rgb_paths) > 0, "{} is empty".format(scene_path)

                # load scene camera and maybe ground truth
                video_metadata = {}
                for json_name in json_names:
                    json_path = osp.join(scene_path, json_name + ".json")
                    if osp.exists(json_path):
                        video_metadata[json_name] = load_json(json_path)
                    else:
                        video_metadata[json_name] = None

                # load masks
                frame_mask_type_paths = {}
                for mask_type in mask_types:
                    mask_paths = Path(scene_path).glob("{}/*.png".format(mask_type))

                    frame_mask_paths = {}
                    for mask_path in mask_paths:
                        frame_id = int(mask_path.stem.split("_")[0])
                        frame_mask_paths.setdefault(frame_id, []).append(mask_path)

                    frame_mask_type_paths[mask_type] = frame_mask_paths

                for idx_frame in range(len(rgb_paths)):
                    # get rgb path
                    rgb_path = rgb_paths[idx_frame]

                    # get id frame
                    frame_id = int(osp.splitext(osp.basename(rgb_path))[0])

                    # get depth path
                    depth_path = osp.join(scene_path, "depth", "{:06d}.{}".format(frame_id, depth_suffix))

                    # add to metadata
                    metadata["scene_id"].append(scene_id)
                    metadata["frame_id"].append(frame_id)
                    metadata["rgb_path"].append(rgb_path)

                    if osp.normpath(depth_path) in depth_paths:
                        metadata["depth_path"].append(depth_path)
                    else:
                        metadata["depth_path"].append(None)

                    for json_name in json_names:
                        data = video_metadata[json_name]
                        if data:
                            metadata[json_name].append(data[str(frame_id)])
                        else:
                            metadata[json_name].append(None)

                    for mask_type in mask_types:
                        data = frame_mask_type_paths[mask_type]
                        if data:
                            metadata[mask_type] = data[frame_id]
                        else:
                            metadata[mask_type] = None

            # casting format of metadata
            metadata = casting_format_to_save_json(metadata)
            save_json(metadata_path, metadata)
        else:
            # load exiting metadata
            metadata = load_json(metadata_path)

        # create panda data frame with metadata
        self.metadata = pd.DataFrame.from_dict(metadata, orient="index")
        self.metadata = self.metadata.transpose()

        # shuffle metadata
        if shuffle_data:
            self.metadata = self.metadata.sample(frac=1, random_state=2024).reset_index()

        finish_time = time.time()
        logging.info("Finish loading metadata of size {} in {:.2f} seconds".format(
            len(self.metadata), finish_time - start_time))

    def __len__(self) -> int:
        return len(self.metadata)


class BaseBOPTest(BaseBOP):
    def __init__(self,
                 root_dir: str,
                 split: str,
                 load_depth: bool = False,
                 depth_scale: float = 1.0,
                 load_gt: bool = False,
                 reset_metadata: bool = True,
                 shuffle_data: bool = False) -> None:
        super().__init__(root_dir, split)

        self.load_depth = load_depth
        self.depth_scale = depth_scale
        self.load_gt = load_gt

        # load scene and meta data
        self.load_list_scene(split=split)
        self.load_metadata(reset_metadata=reset_metadata, split=split, load_gt=self.load_gt)

        # shuffle metadata
        if shuffle_data:
            self.metadata = self.metadata.sample(frac=1, random_state=2024).reset_index()

        self.rgb_transform = T.Compose([T.ToTensor(),
                                        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    def __getitem__(self, idx: int) -> Dict:
        # load rgb
        rgb_path = self.metadata.iloc[idx]["rgb_path"]
        image = self.rgb_transform(Image.open(rgb_path).convert("RGB"))     # 3 x H x W

        # load depth
        if self.load_depth:
            depth_path = self.metadata.iloc[idx]["depth_path"]
            depth = np.array(Image.open(depth_path), dtype=np.float32)      # H x W
            depth /= self.depth_scale                                       # H x W

        # load camera data
        scene_camera = self.metadata.iloc[idx]["scene_camera"]
        cam_intrinsic = np.array(scene_camera["cam_K"]).reshape(3, 3)       # 3 x 3
        depth_scale = scene_camera["depth_scale"]

        scene_id = self.metadata.iloc[idx]["scene_id"]
        frame_id = self.metadata.iloc[idx]["frame_id"]

        result = {"image": image,                   # 3 x H x W
                  "scene_id": scene_id,             # 1
                  "frame_id": frame_id,             # 1
                  "cam_intrinsic": cam_intrinsic}   # 3 x3

        if self.load_depth:
            result["depth"] = depth                 # H x W
            result["depth_scale"] = depth_scale     # 1

        return result


class BOPTemplate(Dataset):
    def __init__(self,
                 template_dir: str,
                 obj_ids: Optional[List[int]],
                 predefined_poses_dir: str,
                 level_templates: int,
                 max_level_templates: int,
                 pose_distribution: str,
                 processing_config: Dict,
                 num_boarding_images_per_obj: int = 50) -> None:
        self.template_dir = template_dir

        if obj_ids is None:
            # search for available object templates
            obj_ids = [int(obj_id[len("obj_"):]) for obj_id in os.listdir(template_dir)
                       if osp.isdir(osp.join(template_dir, obj_id))]
            obj_ids = sorted(set(obj_ids))
            logging.info("Found {} objects in {}".format(obj_ids, self.template_dir))
        self.obj_ids = obj_ids

        if "onboarding_static" in template_dir or "onboarding_dynamic" in template_dir:
            self.model_free_onboarding = True
        else:
            self.model_free_onboarding = False

        # for HOT3D, we have black objects, so we use gray background
        if "hot3d" in template_dir:
            self.use_gray_background = True
            logging.info("Use gray background for HOT3D")
        else:
            self.use_gray_background = False

        self.processing_config = processing_config
        self.rgb_transform = T.Compose([T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        self.proposal_processor = CropResizePad(self.processing_config.image_size,
                                                pad_value=0.5 if self.use_gray_background else 0)

        self.index_templates = load_indices_level_in_level_mapping(poses_dir=predefined_poses_dir,
                                                                   level=level_templates,
                                                                   max_level=max_level_templates,
                                                                   pose_distribution=pose_distribution,
                                                                   return_camera=False)     # T

        self.num_boarding_images_per_obj = num_boarding_images_per_obj    # to avoid memory issue

    def __len__(self) -> int:
        return len(self.obj_ids)

    def _getitem_model_based(self, idx: int):
        templates, masks, boxes = [], [], []
        for id_template in self.index_templates:
            # load image with alpha channel mask
            image = Image.open("{}/obj_{:06d}/{:06d}.png".format(
                self.template_dir, self.obj_ids[idx], id_template))     # H x W x 4

            # extract bounding box
            box = image.getbbox()       # 4 (xyxy)
            if box is None:
                # bounding box can be None, take whole empty image as 'box' then
                box = (0, 0, image.width, image.height)
            boxes.append(box)

            # extract and process mask
            mask = image.getchannel("A")                                # H X W
            mask = torch.from_numpy(np.array(mask) / 255).float()       # H X W
            masks.append(mask.unsqueeze(-1))                            # 1 x H X W

            # process image to rgb
            if self.use_gray_background:
                gray_image = Image.new("RGB", image.size, (128, 128, 128))      # H x W x 3
                gray_image.paste(image, mask=image.getchannel("A"))
                image = gray_image.convert("RGB")       # H x W x 3
            else:
                image = image.convert("RGB")            # H x W x 3
            image = torch.from_numpy(np.array(image) / 255).float()     # H x W x 3
            templates.append(image)

        templates = torch.stack(templates).permute(0, 3, 1, 2)      # B x 3 x H x W
        masks = torch.stack(masks).permute(0, 3, 1, 2)              # B x 1 x H x W
        boxes = torch.from_numpy(np.array(boxes))                   # B x 4

        templates_cropped = self.proposal_processor(images=templates, boxes=boxes)      # B x 3 x H' x W'
        templates_cropped = self.rgb_transform(templates_cropped)                       # B x 3 x H' x W'
        masks_cropped = self.proposal_processor(images=masks, boxes=boxes)              # B x 1 x H' x W'
        masks_cropped = masks_cropped[:, 0]                                             # B x H' x W'

        return {"templates": templates_cropped,     # B x 3 x H' x W'
                "template_masks": masks_cropped}    # B x H' x W'

    def _getitem_model_free(self, idx: int):
        if "onboarding_static" in self.template_dir:
            # static onboarding
            obj_dirs = [osp.join(self.template_dir, "obj_{:06d}_up".format(self.obj_ids[idx])),
                        osp.join(self.template_dir, "obj_{:06d}_down".format(self.obj_ids[idx]))]
            num_selected_images = self.num_boarding_images_per_obj // 2     # split number of image between both views
        else:
            # dynamic onboarding
            obj_dirs = [osp.join(self.template_dir, "obj_{:06d}".format(self.obj_ids[idx]))]
            num_selected_images = self.num_boarding_images_per_obj

        templates, masks, boxes = [], [], []
        for obj_dir in obj_dirs:
            rgb_paths = sorted(Path(obj_dir).glob("rgb/*.[pj][pn][g]"))             # B
            masks_paths = sorted(Path(obj_dir).glob("mask_visib/*.[pj][pn][g]"))    # B

            assert len(rgb_paths) == len(masks_paths), (
                "Found {] object images but only {} corresponding masks, they are not equal.".format(len(rgb_paths), len(masks_paths)))

            # subsample a bit if there are too many images
            selected_indices = np.random.choice(len(rgb_paths), num_selected_images,
                                                replace=len(rgb_paths) < num_selected_images)  # B'

            for index in tqdm(selected_indices):
                # load image and mask
                image = Image.open(rgb_paths[index])    # H x W x 3
                mask = Image.open(masks_paths[index])   # H x W

                # extract bounding box
                box = mask.getbbox()  # 4 (xyxy)
                if box is None:
                    # bounding box can be None, take whole empty image as 'box' then
                    box = (0, 0, image.width, image.height)
                boxes.append(box)

                # mask and process image to rgb
                image = np.array(image.convert("RGB"))                          # H x W x 3
                mask = np.array(mask)                                           # H x W
                masked_image = image * np.expand_dims(mask > 0, -1)             # H x W x 3
                masked_image = torch.from_numpy(masked_image / 255).float()     # H x W x 3
                templates.append(masked_image)

                # process mask
                mask = torch.from_numpy(mask / 255).float()             # H x W
                masks.append(mask.unsqueeze(-1))                        # 1 x H x W

        templates = torch.stack(templates).permute(0, 3, 1, 2)      # B x 3 x H x W
        masks = torch.stack(masks).permute(0, 3, 1, 2)              # B x 1 x H x W
        boxes = torch.tensor(np.array(boxes))                       # B x 4

        templates_cropped = self.proposal_processor(images=templates, boxes=boxes)      # B x 3 x H' x W'
        templates_cropped = self.rgb_transform(templates_cropped)                       # B x 3 x H' x W'
        masks_cropped = self.proposal_processor(images=masks, boxes=boxes)              # B x 1 x H' x W'
        masks_cropped = masks_cropped[:, 0]                                             # B x H' x W'

        return {"templates": templates_cropped,     # B x 3 x H' x W'
                "template_masks": masks_cropped}    # B x H' x W'

    def __getitem__(self, idx: int):
        if self.model_free_onboarding:
            return self._getitem_model_free(idx)
        else:
            return self._getitem_model_based(idx)
