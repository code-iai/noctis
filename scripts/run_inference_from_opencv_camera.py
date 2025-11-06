import os
import os.path as osp
import logging
import argparse

from collections import defaultdict
import itertools

import numpy as np
import cv2
import distinctipy
import torchvision.transforms as T

from src.utils.visualization_detectron2 import NOCTISVisualizer
from src.utils.visualization_utils import visualize_object

from omegaconf import OmegaConf
from hydra import compose, initialize
from hydra.utils import instantiate

from typing_extensions import List, Optional, Union

# set level logging
logging.basicConfig(level=logging.INFO)


def run_inference_and_visualize(device: Union[int, str],
                                config_file_path: Optional[str],
                                config_parameter_overrides: Optional[str],
                                camera_intrinsic: List[float],
                                ignore_object_ids: List[int],
                                minimum_scores: List[float],
                                easy_mode: bool,
                                object_names: List[str], with_labels: bool,
                                jitter_colors: bool,
                                background: str, split_image: bool) -> None:
    logging.info("Starting process to get 'real-time' object segmentation masks of video frames.")

    # load the hydra config file
    if config_file_path:
        config_path, config_name = osp.split(config_file_path)
    else:
        config_path = "../configs"
        config_name = "run_inference"

    with initialize(version_base=None, config_path=config_path, job_name="run_bop23_tests"):
        if config_parameter_overrides is not None:
            config_parameter_overrides = config_parameter_overrides.split(" ")

        cfg = compose(config_name=config_name, overrides=config_parameter_overrides)
    OmegaConf.set_struct(cfg, False)    # allows adding other keys

    # create template dataset
    ref_dataloader_config = cfg.data.reference_dataloader

    if cfg.model.onboarding_config.rendering_type == "onboarding_static":
        logging.info("Using static onboarding images")
        ref_dataloader_config.template_dir += osp.join(cfg.dataset_name, "onboarding_static")
        ref_dataset = instantiate(ref_dataloader_config)
    elif cfg.model.onboarding_config.rendering_type == "onboarding_dynamic":
        logging.info("Using dynamic onboarding images")
        ref_dataloader_config.template_dir += osp.join(cfg.dataset_name, "onboarding_dynamic")
        ref_dataset = instantiate(ref_dataloader_config)
    elif cfg.model.onboarding_config.rendering_type == "templates":
        ref_dataloader_config.template_dir += osp.join(cfg.template_dir, cfg.dataset_name)
        ref_dataset = instantiate(ref_dataloader_config)
    elif cfg.model.onboarding_config.rendering_type == "pbr":
        logging.info("Using BlenderProc for reference images")
        ref_dataloader_config._target_ = "dataloader.bop_pbr.BOPTemplatePBR"
        ref_dataloader_config.root_dir = cfg.data.query_dataloader.root_dir + str(cfg.dataset_name)
        ref_dataloader_config.template_dir += osp.join(cfg.template_dir, cfg.dataset_name)
        ref_dataloader_config.split = "train_pbr"
        os.makedirs(ref_dataloader_config.template_dir, exist_ok=True)  # create, if missing
        ref_dataset = instantiate(ref_dataloader_config)
    else:
        raise NotImplementedError("Unknown template rendering type of name: <{}>".format(cfg.model.onboarding_config.rendering_type))

    # create 'NOCTIS' detector model
    logging.info("Initializing model")
    cfg.model.return_detections = True      # must be 'True' to return the results instead of saving them
    model = instantiate(cfg.model)
    model.dataset_name = cfg.dataset_name
    model.ref_dataset = ref_dataset

    # move model to "cuda", otherwise 'dinov2' does not work
    model.to("cuda")

    # prepare camera
    try:
        # try to cast to int
        device = int(device)
    except ValueError:
        # real string
        pass
    camera = cv2.VideoCapture(device)

    #camera.set(cv2.CAP_PROP_FPS, 20)   # depends on the camera

    logging.info("Initialized model and started usb camera.")

    if ignore_object_ids is None:
        ignore_object_ids = []

    index_to_obj_id = [i for i in ref_dataset.obj_ids if i not in ignore_object_ids]    # filter out unwanted object ids
    obj_id_to_index = {obj_id: i for (i, obj_id) in enumerate(index_to_obj_id)}

    # filter out all detection, whose confidence score is too low, use first 'minimum_scores' as default
    obj_id_minimum_scores = defaultdict(lambda: minimum_scores[0], zip(index_to_obj_id, minimum_scores))

    # use 'object_<id>' as default object name
    if object_names is None:
        object_names = []

    obj_id_obj_names = ["object_{}".format(i) for i in index_to_obj_id]     # object_{:06d} is too long as name
    n = min(len(object_names), len(index_to_obj_id))
    obj_id_obj_names[:n] = object_names[:n]

    obj_id_obj_colors = distinctipy.get_colors(len(index_to_obj_id))

    # camera data
    fx, skew, px, fy, py = camera_intrinsic
    cam_intrinsic = np.array([fx, skew, px, 0.0, fy, py, 0.0, 0.0, 1.0]).reshape(3, 3)  # is not used in RGB only case

    rgb_transform = T.Compose([T.ToTensor(),
                               T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    cv2.namedWindow("NOCTIS", cv2.WINDOW_NORMAL)

    for index in itertools.count():
        # read frame
        ret, frame = camera.read()      # 1, H x W x 3

        if not ret:
            logging.info("Failed to grab frame.")
            cv2.waitKey(50)     # wait, before trying again
            continue

        # stabilise contrast/brightness and convert to rgb image
        cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)    # H x W x 3
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                     # H x W x 3

        # image preprocessing and 'pseudo' batch creation
        batch = {"image": rgb_transform(image)[None],
                 "scene_id": [str(0)],
                 "frame_id": [str(index)],
                 "cam_intrinsic": cam_intrinsic[None]}

        # inference to get detections
        results = model.test_step(batch, index)

        detections = results["detections"]
        if len(detections):
            if split_image:
                rgb = np.copy(image)    # H x W x 3

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
            for (score, object_id, bbox, mask) in zip(detections.scores,
                                                      detections.object_ids,
                                                      detections.boxes,
                                                      detections.masks):
                if object_id in ignore_object_ids or score < obj_id_minimum_scores[object_id]:
                    # filter out all detection, with ignored objects or whose confidence score is too low
                    continue

                object_ids.append(obj_id_to_index[object_id])
                masks.append(mask)      # H x W
                scores.append(score)
                bboxes.append(bbox)     # xyxy

            if len(object_ids):
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
                                           jitter_colors=jitter_colors)     # H x W x 3
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
                                             save_path=None)    # H x W x 3

                if split_image:
                    # combine side-by-side original image with annotated version
                    vis = np.concatenate([rgb, vis], axis=1)    # H x (2*W) x 3

                # convert back to BGR image
                frame = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)    # H x W | (2*W) x 3

        # show frame
        cv2.imshow("NOCTIS", frame)

        if not isinstance(device, int) and camera.get(cv2.CAP_PROP_POS_FRAMES) >= camera.get(cv2.CAP_PROP_FRAME_COUNT):
            camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
            logging.info("Reached video end, restart.")

        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            # ESC pressed
            logging.info("Escape pressed, closing.")
            break

    # clean up
    camera.release()
    cv2.destroyAllWindows()

    logging.info("Finished processing the frames.")
    logging.info("End")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Applies 'NOCTIS' to all video frames.")

    parser.add_argument("-i", "--device", dest="device", type=str, default=0,
                        help="Device/Video file for the video camera.")

    parser.add_argument("-c", "--config_file_path", dest="config_file_path", type=str,
                        help="Path of the hydra configuration file.")
    parser.add_argument("-p", "--config_parameter_overrides", dest="config_parameter_overrides", type=str,
                        help="String of override parameters for hydra configuration file.")

    parser.add_argument("-d", "--camera_intrinsic", dest="camera_intrinsic", type=float, nargs=5, default=[0, 0, 0, 0, 0],
                        help="Camera intrinsic for alle images of the scene in format [fx, skew, px, fy, py].")

    parser.add_argument("-f", "--ignore_object_ids", dest="ignore_object_ids", type=int, nargs="+",
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

    args = parser.parse_args()

    run_inference_and_visualize(device=args.device,
                                config_file_path=args.config_file_path,
                                config_parameter_overrides=args.config_parameter_overrides,
                                camera_intrinsic=args.camera_intrinsic,
                                ignore_object_ids=args.ignore_object_ids,
                                minimum_scores=args.minimum_scores,
                                easy_mode=args.easy_mode,
                                object_names=args.object_names,
                                with_labels=not args.without_labels,
                                jitter_colors=args.jitter_colors,
                                background=args.background,
                                split_image=args.split_image)
