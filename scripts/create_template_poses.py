import blenderproc
import bpy
import bmesh

import os
import os.path as osp
import argparse
import math
import numpy as np


from typing import List, Optional, Tuple

# Use: 'blenderproc run scripts/create_template_poses.py -a -k -l -o "datasets/predefined_poses/ico"'


def get_camera_positions(num_subdiv: int, sort_angles: bool) -> np.ndarray:
    """Construct an icosphere, sub-dived it 'num_subdiv'-times and return the corner positions.

    Note: Based on: "https://github.com/nv-nguyen/cnos/blob/main/src/poses/create_template_poses.py"

    :param int num_subdiv:
    :return: positions list [B, 3]
    :rtype: np.ndarray
    """

    bpy.ops.mesh.primitive_ico_sphere_add(location=(0, 0, 0), enter_editmode=True)
    icos = bpy.context.object
    me = icos.data

    # cut away lower part
    bm = bmesh.from_edit_mesh(me)
    sel = [v for v in bm.verts if v.co[2] < 0]

    bmesh.ops.delete(bm, geom=sel, context="FACES")
    bmesh.update_edit_mesh(me)

    # subdivide and move new vertices out to the surface of the sphere
    for i in range(num_subdiv):
        bpy.ops.mesh.subdivide()

        bm = bmesh.from_edit_mesh(me)
        for v in bm.verts:
            length = math.sqrt(v.co[0] ** 2 + v.co[1] ** 2 + v.co[2] ** 2)
            v.co[0] /= length
            v.co[1] /= length
            v.co[2] /= length
        bmesh.update_edit_mesh(me)

    # cut away zero elevation
    bm = bmesh.from_edit_mesh(me)
    sel = [v for v in bm.verts if v.co[2] <= 0]
    bmesh.ops.delete(bm, geom=sel, context="FACES")
    bmesh.update_edit_mesh(me)

    # convert vertex positions to az,el
    positions = []
    angles = []
    bm = bmesh.from_edit_mesh(me)
    for v in bm.verts:
        x = v.co[0]
        y = v.co[1]
        z = v.co[2]
        az = math.atan2(x, y)  # *180./math.pi
        el = math.atan2(z, math.sqrt(x**2 + y**2))  # *180./math.pi
        # positions.append((az,el))
        angles.append((el, az))
        positions.append((x, y, z))

    bpy.ops.object.editmode_toggle()

    if sort_angles:
        # sort positions, first by az and el
        data = zip(angles, positions)
        positions = sorted(data)
        positions = [y for x, y in positions]

    return np.array(positions)    # B x 3


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


def create_template_poses(num_subdiv: int,
                          sort_angles: bool,
                          origin_position: List[float],
                          scale_factor: float,
                          with_object_poses: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    points = get_camera_positions(num_subdiv=num_subdiv, sort_angles=sort_angles)    # B x 3

    number_poses = points.shape[0]
    cam_poses = np.empty((number_poses, 4, 4))      # B x 4 x 4
    origin_position = np.array(origin_position)

    for i in range(number_poses):
        cam_poses[i] = look_at(points[i], origin_position)  # 4 x 4

    # scale distance to origin, e.g. convert meter to millimeter
    cam_poses[:, :3, 3] *= scale_factor

    if with_object_poses:
        object_poses = inverse_transform(cam_poses)     # B x 4 x 4
    else:
        object_poses = None

    return cam_poses, object_poses      # B x 4 x 4, B x 4 x 4


def create_multiple_level_template_poses(num_max_subdivides: int,
                                         sort_angles: bool,
                                         origin_position: List[float],
                                         scale_factor: float,
                                         sublevel_camera_indices: bool,
                                         sublevel_object_indices: bool,
                                         output_dir: str,
                                         camera_poses_name_template: str,
                                         object_poses_name_template: str):
    with_camera_poses = camera_poses_name_template is not None
    with_object_poses = object_poses_name_template is not None

    max_level = num_max_subdivides

    # create camera and corresponding object poses
    high_level_cam_poses, high_level_object_poses = create_template_poses(num_subdiv=num_max_subdivides,
                                                                          sort_angles=sort_angles,
                                                                          origin_position=origin_position,
                                                                          scale_factor=scale_factor,
                                                                          with_object_poses=with_object_poses)
    max_number_poses = high_level_cam_poses.shape[0]

    # make output dir
    os.makedirs(output_dir, exist_ok=True)

    # save highest level poses
    if with_camera_poses:
        camera_poses_file = osp.join(output_dir, camera_poses_name_template.format(max_level))
        np.save(camera_poses_file, high_level_cam_poses)

        # create a mapping to itself
        if sublevel_camera_indices:
            indices = np.arange(max_number_poses)
            level_to_level_file = osp.join(output_dir, "indices_camera_all_level{}_in_level_{}.npy".format(
                max_level, max_level))
            np.save(level_to_level_file, indices)

    if with_object_poses:
        object_poses_file = osp.join(output_dir, object_poses_name_template.format(max_level))
        np.save(object_poses_file, high_level_object_poses)

        # create a mapping to itself
        if sublevel_object_indices:
            indices = np.arange(max_number_poses)
            level_to_level_file = osp.join(output_dir, "indices_object_all_level{}_in_level_{}.npy".format(
                max_level, max_level))
            np.save(level_to_level_file, indices)

    for level in range(num_max_subdivides-1, -1, -1):
        # create camera and corresponding object poses
        cam_poses, object_poses = create_template_poses(num_subdiv=level,
                                                        origin_position=origin_position,
                                                        sort_angles=sort_angles,
                                                        scale_factor=scale_factor,
                                                        with_object_poses=with_object_poses)

        if with_camera_poses:
            # save lower level camera poses
            camera_poses_file = osp.join(output_dir, camera_poses_name_template.format(level))
            np.save(camera_poses_file, cam_poses)

            # create a mapping from low level to high level
            if sublevel_camera_indices:
                indices = find_nearest_poses_indices(cam_poses, high_level_cam_poses)
                level_to_level_file = osp.join(output_dir, "indices_camera_all_level{}_in_level_{}.npy".format(
                    level, max_level))
                np.save(level_to_level_file, indices)

        if with_object_poses:
            # save lower level object poses
            object_poses_file = osp.join(output_dir, object_poses_name_template.format(level))
            np.save(object_poses_file, object_poses)

            # create a mapping from low level to high level
            if sublevel_object_indices:
                indices = find_nearest_poses_indices(cam_poses, high_level_cam_poses)
                level_to_level_file = osp.join(output_dir, "indices_object_all_level{}_in_level_{}.npy".format(
                    level, max_level))
                np.save(level_to_level_file, indices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creates object and camera poses at different resolutions and mappings between these resolutions.")

    parser.add_argument("-n", "--num_max_subdivides", dest="num_max_subdivides", type=int, default=2,
                        help="Number of maximal subdivides of the icosphere.")

    parser.add_argument("-a", "--sort_angles", dest="sort_angles", action="store_true", default=False,
                        help="Sort the poses based on there angles (altitude, azimuth).")

    parser.add_argument("-p", "--origin_position", dest="origin_position", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        help="Position of the origin/center to move the camera around.")
    parser.add_argument("-f", "--scale_factor", dest="scale_factor", type=float, default=1000.0,
                        help="Scales the distance to the origin.")

    parser.add_argument("-k", "--sublevel_camera_indices", dest="sublevel_camera_indices", action="store_true", default=False,
                        help="Create a mapping from the low level to the high level camera poses.")
    parser.add_argument("-l", "--sublevel_object_indices", dest="sublevel_object_indices", action="store_true", default=False,
                        help="Create a mapping from the low level to the high level object poses.")

    parser.add_argument("-o", "--output_dir", dest="output_dir", type=str, default="predefined_poses",
                        help="Path of the output dir that contains the poses and index maps.")
    parser.add_argument("-c", "--camera_poses_name_template", dest="camera_poses_name_template", type=str, default="camera_poses_level{}.npy",
                        help="Template file name for the created camera poses.")
    parser.add_argument("-d", "--object_poses_name_template", dest="object_poses_name_template", type=str, default="object_poses_level{}.npy",
                        help="Template file name for the created object poses")

    args = parser.parse_args()

    create_multiple_level_template_poses(num_max_subdivides=args.num_max_subdivides,
                                         sort_angles=args.sort_angles,
                                         origin_position=args.origin_position,
                                         scale_factor=args.scale_factor,
                                         sublevel_camera_indices=args.sublevel_camera_indices,
                                         sublevel_object_indices=args.sublevel_object_indices,
                                         output_dir=args.output_dir,
                                         camera_poses_name_template=args.camera_poses_name_template,
                                         object_poses_name_template=args.object_poses_name_template)
