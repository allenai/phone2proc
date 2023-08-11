import numpy as np
from phone2sim.utils.parsing import get_object_pose


def get_object_yaw(rotation: np.ndarray) -> float:
    """Get the yaw of an object from its xyz rotation.

    Assumes x and z are either both 0 or both 180, which is true from the RoomPlan
    API.
    """
    yaw_deg = rotation[1]
    if rotation[0] == 180 and rotation[2] == 180:
        yaw_deg = yaw_deg % 360
        dist_to_270 = abs(yaw_deg - 270)
        dist_to_90 = abs(yaw_deg - 90)
        if dist_to_270 < dist_to_90:
            yaw_deg = yaw_deg + 2 * (270 - yaw_deg)
        elif dist_to_270 > dist_to_90:
            yaw_deg = yaw_deg + 2 * (90 - yaw_deg)
    return yaw_deg


def get_objects(object_files):
    objects = []
    for i, file in enumerate(object_files):
        pose = get_object_pose(file)
        object_type = file.split("/")[-2]

        rect = [
            (pose["bbox"][0] / 2, pose["bbox"][2] / 2),
            (pose["bbox"][0] / 2, -pose["bbox"][2] / 2),
            (-pose["bbox"][0] / 2, -pose["bbox"][2] / 2),
            (-pose["bbox"][0] / 2, pose["bbox"][2] / 2),
        ]
        unity_rotation = get_object_yaw(pose["rotation"])

        room_plan_rotation = pose["rotation"][1]
        if pose["rotation"][0] != 180:
            room_plan_rotation = -room_plan_rotation
        room_plan_rad = room_plan_rotation * np.pi / 180
        for i, p in enumerate(rect):
            rect[i] = (
                p[0] * np.cos(room_plan_rad) - p[1] * np.sin(room_plan_rad),
                p[0] * np.sin(room_plan_rad) + p[1] * np.cos(room_plan_rad),
            )

        # move the objects into position
        for i, p in enumerate(rect):
            rect[i] = (p[0] + pose["position"][0], p[1] + pose["position"][2])

        objects.append(
            {
                "object_type": object_type,
                "bbox": pose["bbox"],
                "rotation": unity_rotation,
                "top_down_rect": rect,
                "position": pose["position"].copy(),
                "file": file,
                "mesh_id": file.split("/")[-1].split(".")[0],
                "room_plan_rotation": room_plan_rotation
            }
        )

    return objects
