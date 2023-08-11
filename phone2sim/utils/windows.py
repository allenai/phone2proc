import copy
import random
from itertools import combinations
from typing import Any, Dict, List, Tuple

import numpy as np
from phone2sim.utils.parsing import get_object_pose
from trimesh import Trimesh

from procthor.databases import asset_database, asset_id_database


def get_wall_holes(
    wall_pose: Dict[str, Any]
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Returns a list of holes in the wall.

    Returns in the format [((x0, y0), (x1, y1)), ...]
    """
    max_point = wall_pose["points"].max(axis=0)
    min_point = wall_pose["points"].min(axis=0)

    points = set([(round(p[0], 3), round(p[1], 3)) for p in wall_pose["points"]])

    # add the corners of the wall incase they are not there
    points.add((round(min_point[0], 3), round(min_point[1], 3)))
    points.add((round(max_point[0], 3), round(max_point[1], 3)))
    points.add((round(min_point[0], 3), round(max_point[1], 3)))
    points.add((round(max_point[0], 3), round(min_point[1], 3)))

    unique_x_points = sorted(list(set([p[0] for p in points])))
    xs_to_ys = {x: set([p[1] for p in points if p[0] == x]) for x in unique_x_points}

    rectangles = []
    for x0, x1 in combinations(unique_x_points, 2):
        x0, x1 = sorted([x0, x1])
        ints = xs_to_ys[x0].intersection(xs_to_ys[x1])
        if len(ints) >= 2:
            for y0, y1 in combinations(ints, 2):
                y0, y1 = sorted([y0, y1])
                rectangles.append(((x0, y0), (x1, y1)))

    mesh = Trimesh(
        vertices=wall_pose["points"],
        faces=np.array(wall_pose["faces"]).reshape(-1, 3),
        process=False,
    )
    mesh.fill_holes()

    holes = []
    eps = 5e-3
    for ((x0, y0), (x1, y1)) in rectangles:
        nearby_points = np.array(
            [
                [x0 + eps, y0 + eps, -0.08],
                [x1 - eps, y0 + eps, -0.08],
                [x1 - eps, y1 - eps, -0.08],
                [x0 + eps, y1 - eps, -0.08],
            ]
        )
        inner_points = mesh.contains(nearby_points)
        if sum(inner_points) == 0:
            holes.append(((x0, y0), (x1, y1)))
    return holes


def get_scale(
    hole: Tuple[Tuple[float, float], Tuple[float, float]], asset_id: str
) -> Dict[str, float]:
    if "Doorway_" in asset_id:
        asset_id = "Doorframe_" + asset_id[len("Doorway_") :]
    (hole_x_min, hole_y_min), (hole_x_max, hole_y_max) = hole
    hole_x_size = hole_x_max - hole_x_min
    hole_y_size = hole_y_max - hole_y_min
    window_x_size = asset_id_database[asset_id]["boundingBox"]["x"]
    window_y_size = asset_id_database[asset_id]["boundingBox"]["y"]
    return {
        "x": hole_x_size / window_x_size,
        "y": hole_y_size / window_y_size,
        "z": 1,
    }


def sample_door_asset(can_open: bool) -> str:
    door_asset_ids = [f"Doorway_{i}" for i in range(1, 11)]
    if can_open:
        door_asset_ids += [f"Doorframe_{i}" for i in range(1, 11)]
    return {
        "assetId": random.choice(door_asset_ids),
        "openness": random.random() * 0.2 + 0.8 if can_open else 0,
    }


def add_windows(
    house,
    wall_file_map,
    wall_holes,
    floor_y,
    ceiling_height,
) -> None:
    house["windows"] = []
    house["doors"] = []
    new_walls = []
    walls_to_remove = set()
    for wall_file, holes in wall_holes.items():
        if len(holes) == 0:
            continue

        front_wall, back_wall = wall_file_map[wall_file]

        if len(holes) > 2:
            holes = [holes[0], holes[1]]
            print(len(holes))
            print(f"More than 2 holes in {wall_file}")

        if len(holes) == 1:
            hole = holes[0]
            (hole_x_min, hole_y_min), (hole_x_max, hole_y_max) = hole
            window_asset_id = "Window_Fixed_60x36"
            is_door = hole_y_min < 1e-2
            if is_door:
                hole_y_min = 0
                asset = sample_door_asset(
                    can_open=front_wall["roomId"] is not None
                    and back_wall["roomId"] is not None
                )
            else:
                asset = {"assetId": window_asset_id}
            scale = get_scale(hole=hole, asset_id=asset["assetId"])

            wall_obj_type = "door" if is_door else "window"
            house[f"{wall_obj_type}s"].append(
                {
                    **asset,
                    "id": wall_obj_type + "|" + wall_file.split("/")[-1].split(".")[0],
                    "holePolygon": [
                        dict(
                            x=hole_x_min,
                            y=hole_y_min,
                            z=0,
                        ),
                        dict(
                            x=hole_x_max,
                            y=hole_y_max,
                            z=0,
                        ),
                    ],
                    "assetPosition": {
                        "x": (hole_x_min + hole_x_max) / 2,
                        "y": (hole_y_min + hole_y_max) / 2,
                        "z": 0,
                    },
                    "room0": front_wall["roomId"],
                    "room1": back_wall["roomId"],
                    "wall0": front_wall["id"],
                    "wall1": back_wall["id"],
                    "scale": scale,
                }
            )

        # NOTE: only assumes 1 split at the moment
        elif len(holes) == 2:
            wall_p0 = front_wall["polygon"][0]
            wall_p1 = front_wall["polygon"][2]

            x_diff = wall_p1["x"] - wall_p0["x"]
            z_diff = wall_p1["z"] - wall_p0["z"]
            angle = np.arctan2(z_diff, x_diff)

            wall_0_length = (holes[0][1][0] + holes[1][0][0]) / 2
            x_split = wall_0_length * np.cos(angle)
            z_split = wall_0_length * np.sin(angle)

            split_point = {
                "x": wall_p0["x"] + x_split,
                "z": wall_p0["z"] + z_split,
            }

            # NOTE: add window 0
            hole = holes[0]
            (hole_x_min, hole_y_min), (hole_x_max, hole_y_max) = hole
            window_asset_id = "Window_Fixed_60x36"
            is_door = hole_y_min < 1e-2
            if is_door:
                hole_y_min = 0
                asset = sample_door_asset(
                    can_open=front_wall["roomId"] is not None
                    and back_wall["roomId"] is not None
                )
            else:
                asset = {"assetId": window_asset_id}
            scale = get_scale(hole=hole, asset_id=asset["assetId"])

            wall_obj_type = "door" if is_door else "window"
            house[f"{wall_obj_type}s"].append(
                {
                    **asset,
                    "id": wall_obj_type
                    + "|"
                    + wall_file.split("/")[-1].split(".")[0]
                    + "_0",
                    "holePolygon": [
                        dict(
                            x=hole_x_min,
                            y=hole_y_min,
                            z=0,
                        ),
                        dict(
                            x=hole_x_max,
                            y=hole_y_max,
                            z=0,
                        ),
                    ],
                    "assetPosition": {
                        "x": (hole_x_min + hole_x_max) / 2,
                        "y": (hole_y_min + hole_y_max) / 2,
                        "z": 0,
                    },
                    "room0": front_wall["roomId"],
                    "room1": back_wall["roomId"],
                    "wall0": front_wall["id"] + "_split_0",
                    "wall1": back_wall["id"] + "_split_0",
                    "scale": scale,
                }
            )

            # NOTE: add window 1
            hole = holes[1]
            (hole_x_min, hole_y_min), (hole_x_max, hole_y_max) = hole
            window_asset_id = "Window_Fixed_60x36"
            is_door = hole_y_min < 1e-2
            if is_door:
                hole_y_min = 0
                asset = sample_door_asset(
                    can_open=front_wall["roomId"] is not None
                    and back_wall["roomId"] is not None
                )
            else:
                asset = {"assetId": window_asset_id}
            scale = get_scale(hole=hole, asset_id=asset["assetId"])

            wall_obj_type = "door" if is_door else "window"
            house[f"{wall_obj_type}s"].append(
                {
                    **asset,
                    "id": wall_obj_type
                    + "|"
                    + wall_file.split("/")[-1].split(".")[0]
                    + "_1",
                    "holePolygon": [
                        dict(
                            x=hole_x_min - wall_0_length,
                            y=hole_y_min,
                            z=0,
                        ),
                        dict(
                            x=hole_x_max - wall_0_length,
                            y=hole_y_max,
                            z=0,
                        ),
                    ],
                    "assetPosition": {
                        "x": (hole_x_min + hole_x_max) / 2 - wall_0_length,
                        "y": (hole_y_min + hole_y_max) / 2,
                        "z": 0,
                    },
                    "room0": front_wall["roomId"],
                    "room1": back_wall["roomId"],
                    "wall0": front_wall["id"] + "_split_1",
                    "wall1": back_wall["id"] + "_split_1",
                    "scale": scale,
                }
            )

            walls_to_remove.add(front_wall["id"])
            walls_to_remove.add(back_wall["id"])

            start_point = front_wall["polygon"][0]
            end_point = front_wall["polygon"][2]

            # NOTE: add first front back
            new_wall_front_1 = copy.deepcopy(front_wall)
            new_wall_front_1["polygon"] = [
                dict(x=start_point["x"], y=floor_y, z=start_point["z"]),
                dict(x=split_point["x"], y=floor_y, z=split_point["z"]),
                dict(x=split_point["x"], y=ceiling_height, z=split_point["z"]),
                dict(x=start_point["x"], y=ceiling_height, z=start_point["z"]),
            ]
            new_wall_front_1["id"] = front_wall["id"] + "_split_0"
            new_walls.append(new_wall_front_1)

            # NOTE: add first wall front
            new_wall_back_1 = copy.deepcopy(back_wall)
            new_wall_back_1["polygon"] = [
                dict(x=split_point["x"], y=floor_y, z=split_point["z"]),
                dict(x=start_point["x"], y=floor_y, z=start_point["z"]),
                dict(x=start_point["x"], y=ceiling_height, z=start_point["z"]),
                dict(x=split_point["x"], y=ceiling_height, z=split_point["z"]),
            ]
            new_wall_back_1["id"] = back_wall["id"] + "_split_0"
            new_walls.append(new_wall_back_1)

            # NOTE: add second wall front
            new_wall_front_2 = copy.deepcopy(front_wall)
            new_wall_front_2["polygon"] = [
                dict(x=split_point["x"], y=floor_y, z=split_point["z"]),
                dict(x=end_point["x"], y=floor_y, z=end_point["z"]),
                dict(x=end_point["x"], y=ceiling_height, z=end_point["z"]),
                dict(x=split_point["x"], y=ceiling_height, z=split_point["z"]),
            ]
            new_wall_front_2["id"] = front_wall["id"] + "_split_1"
            new_wall_front_2["roomId"] = None
            new_walls.append(new_wall_front_2)

            # NOTE: add second wall back
            new_wall_back_2 = copy.deepcopy(back_wall)
            new_wall_back_2["polygon"] = [
                dict(x=end_point["x"], y=floor_y, z=end_point["z"]),
                dict(x=split_point["x"], y=floor_y, z=split_point["z"]),
                dict(x=split_point["x"], y=ceiling_height, z=split_point["z"]),
                dict(x=end_point["x"], y=ceiling_height, z=end_point["z"]),
            ]
            new_wall_back_2["id"] = back_wall["id"] + "_split_1"
            new_walls.append(new_wall_back_2)

            # NOTE: split the front and back walls apart
            # for wall_to_split_id, wall_to_split in [
            #     (wall["id"], wall),
            #     (back_wall_id, back_wall),
            # ]:
            #     start_point = wall_to_split["polygon"][0]
            #     end_point = wall_to_split["polygon"][2]
            #     # if "back" in wall_to_split_id:
            #     # start_point, end_point = end_point, start_point

            #     # NOTE: split wall 0
            #     wall_split_0 = copy.deepcopy(wall_to_split)
            #     wall_split_0["polygon"] = [
            #         dict(x=start_point["x"], y=floor_y, z=start_point["z"]),
            #         dict(x=split_point["x"], y=floor_y, z=split_point["z"]),
            #         dict(x=split_point["x"], y=ceiling_height, z=split_point["z"]),
            #         dict(x=start_point["x"], y=ceiling_height, z=start_point["z"]),
            #     ]
            #     wall_split_0["id"] = wall_to_split["id"] + "_split_0"
            #     new_walls.append(wall_split_0)

            #     # NOTE: split wall 1
            #     wall_split_1 = copy.deepcopy(wall_to_split)
            #     wall_split_1["polygon"] = [
            #         dict(x=split_point["x"], y=floor_y, z=split_point["z"]),
            #         dict(x=end_point["x"], y=floor_y, z=end_point["z"]),
            #         dict(x=end_point["x"], y=ceiling_height, z=end_point["z"]),
            #         dict(x=split_point["z"], y=ceiling_height, z=split_point["z"]),
            #     ]
            #     wall_split_1["id"] = wall_to_split_id + "_split_1"
            #     new_walls.append(wall_split_1)

            #     walls_to_remove.add(wall_to_split_id)

    house["walls"] = [
        wall for wall in house["walls"] if wall["id"] not in walls_to_remove
    ]
    house["walls"] += new_walls
