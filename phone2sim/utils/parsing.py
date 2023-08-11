import os
from typing import Any, Collection, Dict, List

import numpy as np


def get_wall_and_object_files(
    room_usda: str, skip_mesh_ids: Collection[str]
) -> Dict[str, List[str]]:
    base_dir = os.path.dirname(room_usda)
    with open(room_usda, "r") as f:
        lines = f.readlines()

        walls = []
        objects = []

        append_to_walls = False
        append_to_objects = True
        for i, line in enumerate(lines):
            if "Walls_grp" in line:
                append_to_walls = True
                append_to_objects = False
            if "Object_grp" in line:
                append_to_objects = True
                append_to_walls = False

            if ".usda" in line:
                if not append_to_walls and not append_to_objects:
                    raise Exception("No group found")
                line = line[line.find("./") + 2 : line.rfind(".usda") + 5]

                mesh_id = line.split("/")[-1].split(".")[0]
                if mesh_id in skip_mesh_ids:
                    continue

                line = os.path.join(base_dir, line)
                if append_to_walls:
                    walls.append(line)
                if append_to_objects:
                    objects.append(line)
    return {"walls": walls, "objects": objects}


def get_object_pose(
    object_file: str, flip_x: bool = True, flip_y: bool = False
) -> Dict[str, Any]:
    """
    object_file: path to the usda file of the object.
    """
    with open(object_file, "r") as f:
        lines = f.readlines()
        position, bbox, rotation, point_center, faces, normals = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
        for line in lines:
            if "normal3f[] normals" in line:
                assert normals is None
                normals = eval(line.split("=")[1].strip())
            if "int[] faceVertexIndices" in line:
                assert faces is None
                faces = eval(line.split("=")[1].strip())
            if "point3f[] points" in line:
                assert bbox is None
                points = np.array(eval(line.split("=")[1].strip()))
                bbox = points.max(axis=0) - points.min(axis=0)
                point_center = (
                    np.abs(points.max(axis=0)) - np.abs(points.min(axis=0))
                ) / 2
            if "matrix4d xformOp:transform" in line:
                assert rotation is None and position is None
                transform = np.array(eval(line.split("=")[1].strip()))

                position = transform[3, :3]
                position += point_center

                roll = np.arctan2(transform[2, 1], transform[2, 2]) * 180 / np.pi
                pitch = (
                    np.arctan2(
                        transform[2, 0],
                        np.sqrt(transform[2, 1] ** 2 + transform[2, 2] ** 2),
                    )
                    * 180
                    / np.pi
                )
                yaw = np.arctan2(transform[1, 0], transform[0, 0]) * 180 / np.pi
                rotation = np.array([roll, pitch, yaw])
    assert (
        position is not None
        and bbox is not None
        and rotation is not None
        and faces is not None
        and normals is not None
    )
    pose = {
        "position": position.copy(),
        "rotation": rotation.copy(),
        "bbox": bbox.copy(),
        "transform": transform.copy(),
        "points": points.copy(),
        "faces": faces,
        "normals": normals,
    }

    return pose


def get_line_segments_and_walls(wall_files):
    walls = []
    line_segments = []
    for file in wall_files:
        pose = get_object_pose(file)
        walls.append({"file": file, "pose": pose})
        points = np.array(
            [[-pose["bbox"][0] / 2, 0, 0, 1], [pose["bbox"][0] / 2, 0, 0, 1]]
        )
        t_points = pose["transform"].T @ points.T
        x1 = t_points[0, 0]
        z1 = t_points[2, 0]
        x2 = t_points[0, 1]
        z2 = t_points[2, 1]
        line_segments.append([(x1, z1), (x2, z2)])
    return line_segments, walls
