import numpy as np

from detectron2.utils.visualizer import Visualizer, ColorMode, VisImage
from detectron2.utils.visualizer import _create_text_labels, GenericMask
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances

from typing_extensions import List, Tuple, Optional, Sequence


# override the default visualizer
class CustomVisualizer(Visualizer):
    def __init__(self, img_rgb: np.ndarray, metadata: MetadataCatalog, scale: float = 1.0,
                 instance_mode: ColorMode = ColorMode.IMAGE,
                 jitter_colors: bool = False,
                 with_labels: bool = True) -> None:
        super().__init__(img_rgb, metadata, scale, instance_mode)

        self.jitter_colors = jitter_colors
        self.with_labels = with_labels

    def draw_instance_predictions(self, predictions) -> VisImage:
        """Draw instance-level prediction results on an image.

        :param predictions: the output of an instance detection/segmentation model.
                            Following fields will be used to draw: "pred_boxes", "pred_classes", "scores",
                            "pred_masks" (or "pred_masks_rle").
        :return: Image object with visualizations.
        """
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if self.with_labels:
            scores = predictions.scores if predictions.has("scores") else None
            labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        else:
            labels = None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [(self._jitter(self.metadata.thing_colors[c])
                       if self.jitter_colors else self.metadata.thing_colors[c])
                      for c in classes]
            alpha = 0.5
        else:
            colors = None
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.reset_image(
                self._create_grayscale_image((predictions.pred_masks.any(dim=0) > 0).numpy()
                                             if predictions.has("pred_masks") else None))
            alpha = 0.3
        self.overlay_instances(masks=masks,
                               boxes=boxes,
                               labels=labels,
                               keypoints=keypoints,
                               assigned_colors=colors,
                               alpha=alpha)
        return self.output


class NOCTISVisualizer(object):
    def __init__(self, obj_names: List[str], obj_colors: Optional[List[Tuple[int, int, int]]],
                 img_size: Sequence[int], jitter_colors: bool = False, with_labels: bool = True) -> None:
        super(NOCTISVisualizer, self).__init__()

        metadata = self.build_metadata(obj_names, obj_colors)
        self.visualizer = CustomVisualizer(np.zeros((img_size[0], img_size[1], 3)),
                                           metadata=metadata,
                                           scale=1,
                                           instance_mode=ColorMode.SEGMENTATION,
                                           jitter_colors=jitter_colors,
                                           with_labels=with_labels)

    def build_metadata(self, obj_names, obj_colors: Optional[List[Tuple[int, int, int]]] = None) -> MetadataCatalog:
        # define custom metadata
        custom_metadata = MetadataCatalog.get(obj_names[0])     # use the first obj name as the dataset name

        if not hasattr(custom_metadata, "thing_classes"):
            custom_metadata.thing_classes = obj_names

            if obj_colors is None:
                obj_colors = np.random.rand(len(obj_names), 3).tolist()

            custom_metadata.thing_colors = obj_colors

        return custom_metadata

    def convert_to_instances(self, masks: np.ndarray, bboxes: np.ndarray,
                             scores: np.ndarray, labels: np.ndarray) -> Instances:
        """
        :param np.ndarray masks: B x H x W
        :param np.ndarray bboxes: B x 4    (xyxy)
        :param np.ndarray scores: B
        :param np.ndarray labels: B
        """
        # create an Instances object
        instances = Instances(masks.shape[1:])
        instances.set("pred_boxes", bboxes)
        instances.set("pred_masks", masks)

        if scores is not None:
            instances.set("scores", scores)
        if labels is not None:
            instances.set("pred_classes", labels)

        return instances

    def forward(self, rgb: np.ndarray, masks: np.ndarray, bboxes: np.ndarray,
                scores: np.ndarray, labels: np.ndarray, save_path: Optional[str]) -> Optional[np.ndarray]:
        """
        :param np.ndarray rgb: H x W x 3
        :param np.ndarray masks: B x H x W
        :param np.ndarray bboxes: B x 4    (xyxy)
        :param np.ndarray scores: B
        :param np.ndarray labels: B
        :param Optional[str] save_path: If unequal 'None', string path of the output image file, otherwise the image will be returned.
        """
        self.visualizer.output.reset_image(rgb)

        instances = self.convert_to_instances(masks, bboxes, scores, labels)
        self.visualizer.draw_instance_predictions(instances)

        output = self.visualizer.get_output()
        if save_path is not None:
            # save image
            output.save(save_path)
        else:
            # return image
            return output.get_image()
