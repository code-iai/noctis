import os
import os.path as osp
import numpy as np


from typing_extensions import Tuple, Optional, Union


def create_transformation_matrix(rotation: np.ndarray,
                                 translation: np.ndarray,
                                 scale_object: np.number = 1.0,
                                 scale_translation: np.number = 1.0) -> np.ndarray:
    """Combines the rotation matrix and translation vector to transformation matrix.

    :param np.ndarray rotation: [3, 3]
    :param np.ndarray translation: [3]
    :param np.number scale_object: [1]
    :param np.number scale_translation: [1]
    :return: [4, 4] (upper 3x4 matrix part is [R|t])
    :rtype: np.ndarray
    """
    trans_mat = np.eye(4)

    trans_mat[:3, :3] = scale_object * rotation
    trans_mat[:3, 3] = scale_translation * translation

    return trans_mat


def normalize(vec: np.ndarray) -> np.ndarray:
    """Normalize the vector.

    :param np.ndarray vec: [N]
    :return: normalized vector [N]
    :rtype: np.ndarray
    """
    return vec / np.linalg.norm(vec, axis=-1, keepdims=True)


def look_at(from_point: np.ndarray, to_point: np.ndarray, inverted_camera_direction: bool = True) -> np.ndarray:
    """Creates a camera-to-world transformation matrix for a camera looking from a point to another one.

    Note: Based on "https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function/framing-lookat-function.html"

    :param np.ndarray from_point: [3]
    :param np.ndarray to_point: [3]
    :param np.ndarray inverted_camera_direction:
    :return: [4, 4] (upper 3x4 matrix part is [R|t])
    :rtype: np.ndarray
    """
    forward = normalize(from_point - to_point)      # 3

    if inverted_camera_direction:
        # camera points in positive z-axis direction
        forward *= -1

    # use z-axis as helper axis
    up_temp_axis = np.array([0.0, 0.0, 1.0])    # 3

    right = np.cross(up_temp_axis, forward)     # 3
    right_norm = np.linalg.norm(right)          # 3
    if right_norm < 1e-3:
        # camera location is parallel to z-axis, use y-axis instead
        right = normalize(np.cross(np.array([0.0, 1.0, 0.0]), forward))     # 3
    else:
        right /= right_norm                                                 # 3

    if inverted_camera_direction:
        right *= -1

    up = normalize(np.cross(forward, right))    # 3

    # create 4x4 camera-to-world transformation matrix
    twc = np.stack((right, up, forward, from_point), axis=-1)   # 3 x 4
    hom_vec = np.array([[0.0, 0.0, 0.0, 1.0]])                        # 1 x 4
    twc = np.concatenate((twc, hom_vec), axis=-2)               # 4 x 4

    return twc  # 4 x 4


def inverse_transform(poses: np.ndarray) -> np.ndarray:
    """Returns the inverse of transformation matrix.

    :param np.ndarray poses: [B, 4, 4] (upper 3x4 matrix part is [R|t])
    :return: [B, 4, 4]
    :rtype: np.ndarray
    """
    inverted_poses = np.zeros_like(poses)                           # B x 4 x 4
    inverted_rot = np.transpose(poses[:, :3, :3], axes=(0, 2, 1))   # B x 3 x 3
    t = poses[:, :3, 3]                                             # B x 3
    inverted_poses[:, :3, :3] = inverted_rot
    inverted_poses[:, :3, 3] = -np.einsum("bmn, bn -> bm", inverted_rot, t)
    inverted_poses[:, 3, 3] = 1

    return inverted_poses   # B x 4 x 4


def find_nearest_poses_indices(query_poses: np.ndarray, source_poses: np.ndarray) -> np.ndarray:
    """Returns the index of the nearest pose based on the forward direction of the orientation.

    :param np.ndarray  query_poses: [M, 4, 4]   (upper 3x4 matrix part is [R|t])
    :param np.ndarray source_poses: [N, 4, 4]
    :return: [M]
    :rtype: np.ndarray
    """
    # use the forward direction from the corresponding camera transformation matrix
    query_camera_forward = query_poses[:, 2, :3]        # M x 3
    source_camera_forward = source_poses[:, 2, :3]      # N x 3

    distances = np.linalg.norm(query_camera_forward[:, None, :] - source_camera_forward[None, :, :],
                               axis=-1)         # M x N
    indices = np.argmin(distances, axis=-1)     # M

    return indices  # M


def get_obj_poses_from_template_level(poses_dir: str,
                                      level: int = 0,
                                      pose_distribution: str = "all",
                                      return_camera: bool = False,
                                      return_index: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Load the camera or object poses from the corresponding template'.npy' files.

    :param str poses_dir:
    :param int level:
    :param str pose_distribution:
    :param bool return_camera:
    :param str return_index:
    :return loaded poses [B, 4, 4] and optional list of indices, who were not filtered out.
    :rtype: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
    """
    if return_camera:
        poses_path = osp.join(poses_dir, "camera_poses_level{}.npy".format(level))
    else:
        poses_path = osp.join(poses_dir, "object_poses_level{}.npy".format(level))

    # load poses
    poses = np.load(poses_path)     # B x 4 x 4 (upper 3x4 matrix part is [R|t])

    if pose_distribution == "all":
        if return_index:
            return poses, np.arange(len(poses))
        else:
            return poses
    elif pose_distribution == "upper":
        filter_indices = poses[:, 2, 3] >= 0
        filtered_poses = poses[filter_indices]

        if return_index:
            return filtered_poses, np.arange(len(poses))[filter_indices]
        else:
            return filtered_poses
    elif pose_distribution == "under":
        filter_indices = poses[:, 2, 3] < 0
        filtered_poses = poses[filter_indices]

        if return_index:
            return filtered_poses, np.arange(len(poses))[filter_indices]
        else:
            return filtered_poses
    else:
        raise ValueError("The value of 'pose_distribution' is <{}> unknown.".format(pose_distribution))


def load_indices_level_in_level_mapping(poses_dir: str,
                                        level: int = 0,
                                        max_level: int = 2,
                                        pose_distribution: str = "all",
                                        return_camera: bool = False) -> np.ndarray:
    """Load the indices for the mapping from lower level poses to the corresponding poses in the higher levels.

    :param str poses_dir:
    :param int level:
    :param int max_level:
    :param str pose_distribution:
    :param bool return_camera:
    :return: [B]
    :rtype: np.ndarray
    """
    if return_camera:
        index_path = osp.join(poses_dir, "indices_camera_{}_level{}_in_level_{}.npy".format(pose_distribution,
                                                                                            level, max_level))
    else:
        index_path = osp.join(poses_dir, "indices_object_{}_level{}_in_level_{}.npy".format(pose_distribution,
                                                                                            level, max_level))

    return np.load(index_path)      # B
