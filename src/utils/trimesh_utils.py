import numpy as np
import trimesh


def load_mesh(path, ORIGIN_GEOMETRY="BOUNDS"):
    mesh = as_mesh(trimesh.load(path))
    if ORIGIN_GEOMETRY == "BOUNDS":
        AABB = mesh.bounds
        center = np.mean(AABB, axis=0)
        mesh.vertices -= center
    return mesh


def get_obj_diameter(mesh_path):
    mesh = load_mesh(mesh_path)
    extents = mesh.extents * 2
    return np.linalg.norm(extents)


def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        result = trimesh.util.concatenate(
            [
                trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
                for m in scene_or_mesh.geometry.values()
            ]
        )
    else:
        result = scene_or_mesh
    return result
