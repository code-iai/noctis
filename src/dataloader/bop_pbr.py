import logging
import os
import os.path as osp
from tqdm import tqdm
import time
from pathlib import Path
from PIL import Image

import numpy as np
import torch
import torchvision.transforms as T
import pandas as pd

from src.dataloader.bop import BaseBOP
import src.poses.utils as pose_utils
from src.utils.bbox_utils import CropResizePad
from src.utils.inout import load_json, save_json, casting_format_to_save_json


from typing_extensions import Dict, List, Optional, Union, Tuple


class BOPTemplatePBR(BaseBOP):
    def __init__(self,
                 root_dir: str,
                 split: str,
                 template_dir: str,
                 obj_ids: Optional[List[int]],
                 predefined_poses_dir: str,
                 level_templates: int,
                 max_level_templates: int,
                 pose_distribution: str,
                 processing_config: Dict,
                 min_visib_fract=0.8,
                 max_num_scenes=10,
                 max_num_frames=1000,
                 max_num_objects_poses: int = 5000,
                 reset_metadata: bool = True,
                 shuffle_data: bool = False) -> None:
        self.root_dir = root_dir
        self.split = split

        self.template_dir = template_dir

        if obj_ids is None:
            # search for available object templates
            obj_ids = [int(obj_id[len("obj_"):]) for obj_id in os.listdir(template_dir)
                       if osp.isdir(osp.join(template_dir, obj_id))]
            obj_ids = sorted(set(obj_ids))
            logging.info("Found {} objects in {}".format(obj_ids, self.template_dir))
        self.obj_ids = obj_ids

        self.predefined_poses_dir = predefined_poses_dir
        self.level_templates = level_templates
        self.max_level_templates = max_level_templates
        self.pose_distribution = pose_distribution

        self.template_poses = pose_utils.get_obj_poses_from_template_level(poses_dir=self.predefined_poses_dir,
                                                                           level=self.level_templates,
                                                                           pose_distribution=self.pose_distribution,
                                                                           return_camera=False)     # T x 4 x 4

        self.list_scenes = None
        self.metadata = None

        # load scene
        self.load_list_scene(split=split)

        # not need to search all scenes and frames since it is slow
        self.list_scenes = self.list_scenes[:max_num_scenes]
        logging.info("Found {} scene, but using only {} scene for faster runtime".format(len(
            self.list_scenes), max_num_scenes))

        self.min_visib_fract = min_visib_fract
        self.max_num_frames = max_num_frames
        self.max_num_objects_poses = max_num_objects_poses

        # load meta data
        self.load_processed_metadata(reset_metadata=reset_metadata, split=split, shuffle_data=shuffle_data)

        self.processing_config = processing_config
        self.rgb_transform = T.Compose([T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        self.proposal_processor = CropResizePad(self.processing_config.image_size)

    def __len__(self) -> int:
        return len(self.obj_ids)

    def load_metadata(self, reset_metadata: bool = True, split: str = "train_pbr", shuffle_data: bool = False) -> None:
        start_time = time.time()

        logging.info("Loading metadata for split {}".format(split))

        metadata_path = osp.join(self.root_dir, "{}_metadata.json".format(split))

        if reset_metadata or not osp.exists(metadata_path):
            # make new meta data
            metadata = {v: [] for v in ["scene_id", "frame_id", "rgb_path", "visib_fract", "obj_id", "idx_obj", "obj_poses"]}

            for scene_path in tqdm(self.list_scenes, desc="Loading metadata"):
                scene_id = scene_path.split("/")[-1]

                if osp.exists(osp.join(scene_path, "rgb")):
                    rgb_paths = sorted(Path(scene_path).glob("rgb/*.[pj][pn][g]"))
                else:
                    rgb_paths = sorted(Path(scene_path).glob("gray/*.tif"))

                rgb_paths = [str(x) for x in rgb_paths]

                # load poses
                scene_gt_info = load_json(osp.join(scene_path, "scene_gt_info.json"))
                scene_gt = load_json(osp.join(scene_path, "scene_gt.json"))

                for idx_frame in range(len(rgb_paths)):
                    # get rgb path
                    rgb_path = rgb_paths[idx_frame]

                    # get frame id
                    frame_id = int(osp.splitext(osp.basename(rgb_path))[0])

                    # get object ids and ground truth poses and ground truth visibility fractions
                    obj_ids = [int(x["obj_id"]) for x in scene_gt[str(frame_id)]]   # B
                    obj_poses = [pose_utils.create_transformation_matrix(np.reshape(x["cam_R_m2c"], (3, 3)),
                                                                         np.reshape(x["cam_t_m2c"], 3))
                                 for x in scene_gt[str(frame_id)]]  # B x 4 x 4
                    visib_fracts = [float(x["visib_fract"]) for x in scene_gt_info[str(frame_id)]]  # B

                    # add to metadata
                    num_objects = len(obj_ids)
                    metadata["scene_id"].extend([scene_id] * num_objects)
                    metadata["frame_id"].extend([frame_id] * num_objects)
                    metadata["rgb_path"].extend([rgb_path] * num_objects)

                    metadata["visib_fract"].extend(visib_fracts)
                    metadata["obj_id"].extend(obj_ids)
                    metadata["idx_obj"].extend(range(num_objects))
                    metadata["obj_poses"].extend(obj_poses)

                    if idx_frame > self.max_num_frames:
                        break

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

    def load_processed_metadata(self, reset_metadata: bool = True, split: str = "train_pbr", shuffle_data: bool = False) -> None:
        metadata_path = osp.join(self.root_dir, "{}_processed_metadata.csv".format(split))

        if reset_metadata or not osp.exists(metadata_path):
            # load template poses to search for in the training data
            template_poses = pose_utils.get_obj_poses_from_template_level(poses_dir=self.predefined_poses_dir,
                                                                          level=self.level_templates,
                                                                          pose_distribution=self.pose_distribution)     # T x 4 x 4

            # load meta data
            self.load_metadata(reset_metadata=reset_metadata, split=split, shuffle_data=shuffle_data)
            init_size = len(self.metadata)

            logging.info("Start processing metadata with initial size {}".format(init_size))

            # filter out objects that are not visible enough
            self.metadata = self.metadata.iloc[np.array(self.metadata["visib_fract"] > self.min_visib_fract)]
            self.metadata = self.metadata.reset_index(drop=True)

            selected_indices = []
            dataframe_indices = np.arange(0, len(self.metadata))

            # for each object, find reference frames by taking top k frames with farthest distance
            for obj_id in tqdm(self.obj_ids, desc="Finding nearest frame close to template poses"):
                # indices of all metadata entries for the current object id
                selected_obj_indices = dataframe_indices[self.metadata["obj_id"] == obj_id]     # B'

                # subsample a bit if there are too many frames
                selected_obj_indices = np.random.choice(selected_obj_indices, self.max_num_objects_poses,
                                                        replace=len(selected_obj_indices) < self.max_num_objects_poses)   # S

                obj_poses = np.stack(self.metadata.iloc[selected_obj_indices].obj_poses, axis=0)    # S x 4 x 4

                # normalize translation to have unit norm/be on the unit S02 sphere
                distances = np.linalg.norm(obj_poses[:, :3, 3], axis=1, keepdims=True)   # S x 1
                obj_poses[:, :3, 3] = obj_poses[:, :3, 3] / distances                    # S x 4 x 4

                # find index of the nearest template pose in the object poses
                nearest_pose_indices = pose_utils.find_nearest_poses_indices(query_poses=template_poses,
                                                                             source_poses=obj_poses)        # T

                # update metadata
                selected_indices.extend(selected_obj_indices[nearest_pose_indices])

            self.metadata = self.metadata.iloc[selected_indices]
            self.metadata = self.metadata.reset_index(drop=True)
            logging.info("Finish processing metadata from {} to {}".format(init_size, len(self.metadata)))

            # save meta data
            self.metadata.to_csv(metadata_path)
        else:
            # load exiting metadata
            self.metadata = pd.read_csv(metadata_path).reset_index(drop=True)

    def __getitem__(self, idx: int):
        templates, masks, boxes = [], [], []
        obj_ids = []
        obj_indices = range(idx * len(self.template_poses), (idx+1) * len(self.template_poses))

        for i in obj_indices:
            # load rgb
            image = Image.open(self.metadata.iloc[i]["rgb_path"])       # H x W x 3

            # load mask
            idx_obj = self.metadata.iloc[i]["idx_obj"]
            scene_id = self.metadata.iloc[i]["scene_id"]
            frame_id = self.metadata.iloc[i]["frame_id"]
            mask_path = osp.join(self.root_dir, self.split, "{:06d}".format(int(scene_id)),
                                 "mask_visib", "{:06d}_{:06d}.png".format(frame_id, idx_obj))
            mask = Image.open(mask_path)    # H x W

            # extract bounding box
            box = mask.getbbox()        # 4 (xyxy)
            if box is None:
                # bounding box can be None, take whole empty image as 'box' then
                box = (0, 0, image.width, image.height)
            boxes.append(box)

            # mask and process image to rgb
            image = np.array(image.convert("RGB"))                          # H x W x 3
            mask = np.array(mask)                                           # H x W
            masked_image = image * np.expand_dims(mask > 0, -1)             # H x W  x 3
            masked_image = torch.from_numpy(masked_image / 255).float()     # H x W  x 3
            templates.append(masked_image)

            # process mask
            mask = torch.from_numpy(mask / 255).float()             # H x W
            masks.append(mask.unsqueeze(-1))

            # check if object is the same the whole loop
            obj_ids.append(self.metadata.iloc[i].obj_id)

        assert len(np.unique(obj_ids)) == 1, "Only support one object per batch but found {}".format(np.unique(obj_ids))

        templates = torch.stack(templates).permute(0, 3, 1, 2)      # B x 3 x H x W
        masks = torch.stack(masks).permute(0, 3, 1, 2)              # B x 1 x H x W
        boxes = torch.from_numpy(np.array(boxes))                   # B x 4

        templates_cropped = self.proposal_processor(images=templates, boxes=boxes)  # B x 3 x H' x W'
        templates_cropped = self.rgb_transform(templates_cropped)                   # B x 3 x H' x W'
        masks_cropped = self.proposal_processor(images=masks, boxes=boxes)          # B x 1 x H' x W'
        masks_cropped = masks_cropped[:, 0]                                         # B x H' x W'

        return {"templates": templates_cropped,     # B x 3 x H' x W'
                "template_masks": masks_cropped}    # B x H' x W'
