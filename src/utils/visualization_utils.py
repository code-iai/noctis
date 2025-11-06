import numpy as np
import cv2

from typing_extensions import Union, Sequence, Tuple


def jitter_color(color: Sequence[float], strength: float = 0.5) -> Tuple[float]:
    """
    Creates a slightly different color compared to the given one via a random modification.

    :param Sequence[float] color: A tuple/list of 3 elements representing an RGB color value with value range [0, 1].
    :param float strength:
    :return: [3]
    """
    noise = np.random.rand(3)					# 3

    # normalize the vector
    noise = noise / np.linalg.norm(noise)		# 3

    jitt_color = np.clip(color + noise * strength, 0.0, 1.0)  # 3

    return tuple(jitt_color)  # 3


def draw_masks(img: np.ndarray, masks: np.ndarray, colors: np.ndarray, alphas: Union[float, np.ndarray] = 0.33)\
        -> np.ndarray:
    """Draws all masks on the given image.

    :param np.ndarray img: Image [H, W, C] to draw the masks on.
    :param np.ndarray masks: Binary masks [B, H, W] to draw.
    :param np.ndarray colors: Colors [B, C] of the masks.
    :param Union[float, np.ndarray] alphas: Alpha [B] values for the masks.
    :return: Image [H, W, C] with masks drawn on.
    :rtype: np.ndarray
    """
    if len(masks.shape) == 2:
        # add batch dimension
        masks = masks[None, ...]        # B x H x W
    if len(colors.shape) == 1:
        # add batch dimension
        colors = colors[None, ...]      # 1 x C

        # same color for all masks
        colors = np.tile(colors, reps=(masks.shape[0], 1))  # B x C

    alphas = np.array(alphas)
    if len(alphas.shape) == 0:
        # scalar
        alphas = alphas * np.ones(masks.shape[0])

    vis = np.copy(img)  # H x W x C

    for (mask, color, alpha) in zip(masks, colors, alphas):
        vis[mask, :] = alpha * color + (1-alpha) * img[mask, :]     # H x W X C

    return vis  # H x W x C


def draw_contours(img: np.ndarray, masks: np.ndarray, colors: np.ndarray, kernel_size: int = 2) -> np.ndarray:
    """Draws the contours around each mask on the given image.

    :param np.ndarray img: Image [H, W, C] to draw contours on.
    :param np.ndarray masks: Binary masks [B, H, W] to draw a contour around.
    :param np.ndarray colors: Colors [B, C] of the contours.
    :param int kernel_size: Size of the dilation kernel.
    :return: Image [H, W, C] with object contours drawn on.
    :rtype: np.ndarray
    """
    if len(masks.shape) == 2:
        # add batch dimension
        masks = masks[None, ...]        # B x H x W
    if len(colors.shape) == 1:
        # add batch dimension
        colors = colors[None, ...]      # 1 x C

        # same color for all contours
        colors = np.tile(colors, reps=(masks.shape[0], 1))  # B x C
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)    # 2

    vis = np.copy(img)  # H x W x C

    for (mask, color) in zip(masks, colors):
        # canny edge detection
        edge = cv2.Canny(image=mask.astype(np.uint8)*255, threshold1=25, threshold2=51)     # H x W

        # dilate edge to increase thickness
        edge = cv2.dilate(src=edge, kernel=np.ones(kernel_size)).astype(bool)               # H x W

        vis[edge, :] = color    # H x W x C

    return vis  # H x W x C


def visualize_object(img: np.ndarray, masks: np.ndarray, labels: np.ndarray, colors: np.ndarray, jitter_colors: bool) \
        -> np.ndarray:
    """Draws all masks and there contours on the given image.

    :param np.ndarray img: Image [H, W, C] to draw contours on.
    :param np.ndarray masks: Binary masks [B, H, W] to draw a contour around.
    :param np.ndarray labels: Class label [B] of each mask.
    :param np.ndarray colors: RGB color [L, 3] for each object class. Values from [0, 1]
    :param bool jitter_colors: If 'True', jitters the color for each instance,
                               if multiple instance of the same class are in the given image. Otherwise,
                               each instance uses the same color.
    :return: Image [H, W, C] with object contours drawn on.
    :rtype: np.ndarray
    """
    # pick the label based color for each mask
    obj_colors = [(jitter_color(colors[l]) if jitter_colors else colors[l]) for l in labels]

    # scale color from [0, 1] float to [0, 255] integer
    obj_colors = np.asarray([(255 * np.asarray(color)).astype(int) for color in obj_colors])

    vis = np.copy(img)

    # draw masks
    vis = draw_masks(img=vis, masks=masks, colors=obj_colors)

    # draw contours
    vis = draw_contours(img=vis, masks=masks, colors=obj_colors)

    return vis
