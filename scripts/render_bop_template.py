import os
import os.path as osp
import subprocess
import numpy as np
from tqdm import tqdm
import time
from omegaconf import DictConfig, OmegaConf
from functools import partial
import multiprocessing
import logging
import hydra
import glob

from src.poses.utils import get_obj_poses_from_template_level

from typing_extensions import List


# set level logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def call_render(idx_obj: int,
                list_cad_path: List[str],
                list_output_dir: List[str],
                obj_pose_path: str,
                disable_output: bool,
                gpus_devices: str,
                renderer: str,
                recenter: bool,
                with_depth: bool):
    output_dir = list_output_dir[idx_obj]
    cad_path = list_cad_path[idx_obj]

    if os.path.exists(output_dir):
        # remove first to avoid overlapping
        os.system("rm -r {}".format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if renderer == "pyrender":
        command = ["python", "-m", "scripts.pyrender_custom",
                   cad_path, obj_pose_path, output_dir, gpus_devices]
    elif renderer == "blenderproc":
        command = ["blenderproc", "run", "./scripts/blenderproc_custom.py",
                   cad_path, obj_pose_path, output_dir, gpus_devices]
    else:
        raise ValueError("Unknown renderer: {}".format(renderer))

    if disable_output:
        command.append("--disable_output")
    if recenter:
        command.append("--recenter")
    if with_depth:
        command.append("--with_depth")
    logger.info("Executing command:\n{}".format(command))
    logger.info("Render for model <{}>.".format(cad_path))
    subprocess.run(command, check=True)

    # make sure the number of rendered images is correct
    num_images = len(glob.glob(f"{output_dir}/*.png"))
    if with_depth:
        num_images //= 2
    if num_images == len(np.load(obj_pose_path)):
        return True
    else:
        logger.info("Found only {} for {} {}".format(num_images, cad_path, obj_pose_path))
        return False


@hydra.main(version_base=None,
            config_path="../configs",
            config_name="download")
def render(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    root_save_dir = osp.join(cfg.data.root_dir, cfg.template_dir)
    template_poses = get_obj_poses_from_template_level(poses_dir=cfg.data.reference_dataloader.predefined_poses_dir,
                                                       level=cfg.level,
                                                       pose_distribution="all",
                                                       return_camera=False)
    template_poses[:, :3, 3] *= 0.4  # zoom to object

    bop23_datasets = ["lmo",
                      "tless",
                      "tudl",
                      "icbin",
                      "itodd",
                      "hb",
                      "ycbv"]
    if cfg.dataset_name is None:
        datasets = bop23_datasets
    else:
        datasets = [cfg.dataset_name]
    for dataset_name in datasets:
        logging.info("Rendering templates for {}".format(dataset_name))

        dataset_save_dir = osp.join(root_save_dir, dataset_name)
        os.makedirs(dataset_save_dir, exist_ok=True)

        if cfg.separated_poses_per_object:
            obj_pose_dir = osp.join(dataset_save_dir, "object_poses")
            os.makedirs(obj_pose_dir, exist_ok=True)
        else:
            obj_pose_path = osp.join(dataset_save_dir, "template_poses.npy")
            np.save(obj_pose_path, template_poses)

        cad_file_ext = ".ply"
        if dataset_name in ["tless"]:
            cad_dir = os.path.join(cfg.data.root_dir, dataset_name, "models/models_cad")
            if not os.path.exists(cad_dir):
                cad_dir = os.path.join(cfg.data.root_dir, dataset_name, "models_cad")
        elif dataset_name in ["hot3d"]:
            cad_file_ext = ".glb"
            cad_dir = os.path.join(cfg.data.root_dir, dataset_name, "object_models")
            if not os.path.exists(cad_dir):
                raise Exception("CAD dir not found at <{}>".format(cad_dir))
        else:
            cad_dir = os.path.join(cfg.data.root_dir, dataset_name, "models/models")
            if not os.path.exists(cad_dir):
                cad_dir = os.path.join(cfg.data.root_dir, dataset_name, "models")
        cad_paths = []
        output_dirs = []
        object_ids = [int(name[4:][:-4]) for name in os.listdir(cad_dir) if name.endswith(cad_file_ext)]
        object_ids = sorted(set(object_ids))

        logger.info("Number of objects found: {}".format(len(object_ids)))

        for object_id in object_ids:
            cad_paths.append(osp.join(cad_dir, "obj_{:06d}{}".format(object_id, cad_file_ext)))
            output_dirs.append(osp.join(dataset_save_dir, "obj_{:06d}".format(object_id)))

            if cfg.separated_poses_per_object:
                obj_pose_path = osp.join(dataset_save_dir, "object_poses", "{:06d}.npy".format(object_id))
                np.save(obj_pose_path, template_poses)

        os.makedirs(dataset_save_dir, exist_ok=True)

        logger.info("Start rendering for {} objects".format(len(cad_paths)))

        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus
        start_time = time.time()
        call_render_with_index = partial(call_render,
                                         list_cad_path=cad_paths,
                                         list_output_dir=output_dirs,
                                         obj_pose_path=obj_pose_path,
                                         disable_output=cfg.disable_output,
                                         gpus_devices=cfg.gpus,
                                         renderer=cfg.renderer,
                                         recenter=cfg.recenter_objects,
                                         with_depth=cfg.with_depth)

        num_workers = int(cfg.num_workers)
        if num_workers > 0:
            # with multiprocessing
            pool = multiprocessing.Pool(processes=num_workers)
            values = list(tqdm(pool.imap_unordered(call_render_with_index, range(len(object_ids))),
                               total=len(object_ids)))
        else:
            # without multiprocessing
            values = list(tqdm(map(call_render_with_index, range(len(object_ids))),
                               total=len(object_ids)))
        logger.info("Finished correctly for {}/{} objects".format(sum(values), len(cad_paths)))
        finish_time = time.time()
        logging.info("Total time to render templates for {}: {}".format(dataset_name, finish_time - start_time))


if __name__ == "__main__":
    render()
