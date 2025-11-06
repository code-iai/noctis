import blenderproc as bproc
import numpy as np
import trimesh
import argparse
import os
import os.path as osp
import sys
from PIL import Image
import logging

from typing import Optional

#import pydevd_pycharm
#pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)


# uses GigaPose "blenderproc.py" as basis, updates it with CNOS "pyrender.py" interfaces changes
# and uses SAM-6D "render_bop_templates.py" to fix the 'texture is always black' bug
def render_blender_proc(cad_path: str,
                        scale: float,
                        obj_poses: np.ndarray,
                        overwrite_color: Optional[np.ndarray],
                        ambient_light: bool,
                        light_energy: float,
                        light_scale: float,
                        img_size: np.ndarray,
                        intrinsic: np.ndarray,
                        output_dir: str,
                        with_depth: bool):
    bproc.init()

    # set camera fixed at 'np.eye(4)'
    cam2world = bproc.math.change_source_coordinate_frame_of_transformation_matrix(
        np.eye(4), ["X", "-Y", "-Z"]
    )
    bproc.camera.add_camera_pose(cam2world)
    bproc.camera.set_intrinsics_from_K_matrix(intrinsic, img_size[1], img_size[0])

    # set light
    if ambient_light:
        light_locations = []
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [0]:   #[0, 1]:    # only light at camera level reduces reflection
                    light_locations.append([x*light_scale, y*light_scale, z*light_scale])
    else:
        # no 'pyrender.SpotLight' because the creates shadows at the cone area
        light_locations = [[0, 0, -light_scale]]

    for location in light_locations:
        light = bproc.types.Light()
        light.set_type("POINT")
        light.set_location(location)
        light.set_energy(light_energy)

    # load object
    obj = bproc.loader.load_obj(cad_path)[0]
    #obj.set_scale([scale, scale, scale])   # does not do anything

    # cannot scale model so scale transformation
    obj_poses[:, :3, :3] *= scale

    if overwrite_color is not None:
        mat = obj.get_materials()[0]
        overwrite_color = np.concatenate([overwrite_color, np.array([1.0])], axis=0)    # add the alpha channel to the color
        mat.set_principled_shader_value("Base Color", overwrite_color)

    # configurate renderer
    # set the amount of samples, which should be used for the color rendering
    bproc.renderer.set_max_amount_of_samples(100)
    # use alpha channel in rgb image output
    bproc.renderer.set_output_format(enable_transparency=True)
    # activate normal and distance rendering
    bproc.renderer.enable_distance_output(True)
    # activate depth rendering
    bproc.renderer.enable_depth_output(activate_antialiasing=False)

    for idx_frame, obj_pose in enumerate(obj_poses):
        #print("render frame {} of {}.".format(idx_frame, len(obj_poses)))
        # update object position in the world
        obj.set_local2world_mat(obj_pose)

        # render as rgba image with depth
        data = bproc.renderer.render()
        data.update(bproc.renderer.render_segmap(map_by="class", use_alpha_channel=True))

        # rgba image
        rgba = Image.fromarray(data["colors"][0].astype(np.uint8))
        rgba.save(osp.join(output_dir, "{:06d}.png".format(idx_frame)))

        if with_depth:
            # masked depth image
            mask = np.array(rgba.getchannel("A"), dtype=bool)
            depth = data["depth"][0] * 1000.0  # convert to mm
            depth[~mask] = 0
            depth = Image.fromarray(depth.astype(np.uint16))
            depth.save(osp.join(output_dir, "{:06d}_depth.png".format(idx_frame)))


#TODO has problems with .glb model files (model is rotated?) from the hot3d dataset
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("cad_path", type=str, help="Path to the model file")
    parser.add_argument("obj_pose", type=str, help="Path to the object pose file")
    parser.add_argument("output_dir", type=str, help="Path to where the final files will be saved")
    parser.add_argument("gpus_devices", type=str, nargs="?", default="cpu", help="CPU/GPU devices for rendering")
    parser.add_argument("--disable_output", action="store_true", default=False, help="Disable output of blender")
    parser.add_argument("--ambient_light", action="store_true", default=False, help="Surrounding light for the object")
    parser.add_argument("--light_energy", type=float, nargs="?", default=50, help="Light Energy")
    parser.add_argument("--light_scale", type=float, nargs="?", default=1.0, help="Scales the light distance")
    parser.add_argument("--radius", type=float, nargs="?",  default=1.0, help="Distance from camera to object")
    parser.add_argument("--not_recenter", action="store_true", default=False, help="Don't recenter the objects at the origin")
    parser.add_argument("--with_depth", action="store_true", default=False, help="Saves the depth images")
    args = parser.parse_args()

    print(args)

    # create output folder
    os.makedirs(args.output_dir, exist_ok=True)

    # prepare cuda acceleration
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus_devices)
    os.environ["EGL_VISIBLE_DEVICES"] = str(args.gpus_devices)

    # load camera pose and transform
    poses = np.load(args.obj_pose)
    poses[:, :3, 3] /= 1000.0   # to meter

    if args.radius != 1:
        poses[:, :3, 3] = poses[:, :3, 3] * args.radius

    # camera intrinsic
    if "tless" in args.output_dir:
        intrinsic = np.array([1075.65091572, 0.0, 360, 0.0, 1073.90347929, 270, 0.0, 0.0, 1.0]).reshape(3, 3)
        img_size = np.array([480, 640])#np.array([540, 720])
        overwrite_color = np.array([0.4, 0.4, 0.4])    # use uniform grey color for the mesh
    else:
        intrinsic = np.array([[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]])
        img_size = np.array([480, 640])
        overwrite_color = None

    # load mesh and automatically concatenate a scene (multiple meshes) to one mesh
    mesh = trimesh.load_mesh(args.cad_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    diameter = np.linalg.norm(mesh.extents)
    if diameter > 50:
        # object is in mm so rescale mesh vertices to meter
        if diameter > 400:
            # very large object
            scale = 0.0005
        else:
            scale = 0.001
    else:
        # object is in meter
        scale = 1.0

    if not args.not_recenter:
        # re-center object at the origin
        re_center_transform = np.eye(4)
        re_center_transform[:3, 3] = -mesh.bounding_box.centroid * scale    # scale is needed here
        print("Object center at {}".format(mesh.bounding_box.centroid))
        poses = np.matmul(poses, re_center_transform)

    if args.disable_output:
        # redirect output to log file
        logfile = os.path.join(args.output_dir, "render.log")
        open(logfile, "a").close()
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)

    render_blender_proc(cad_path=args.cad_path,
                        scale=scale,
                        obj_poses=poses,
                        overwrite_color=overwrite_color,
                        ambient_light=args.ambient_light,
                        light_energy=args.light_energy,
                        light_scale=args.light_scale,
                        intrinsic=intrinsic,
                        img_size=img_size,
                        output_dir=args.output_dir,
                        with_depth=args.with_depth)

    if args.disable_output:
        # disable output redirection
        os.close(1)
        os.dup(old)
        os.close(old)
        os.system("rm {}".format(logfile))
