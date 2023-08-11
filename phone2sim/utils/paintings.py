import random
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ai2thor.controller import Controller
from shapely.geometry import LineString, MultiLineString, Polygon

from procthor.databases import asset_database


def filter_wall_lines(
    wall_lines: List[LineString], min_painting_width: float
) -> List[LineString]:
    filtered_lines = []
    for line in wall_lines:
        if line.length > min_painting_width:
            filtered_lines.append(line)
    return filtered_lines


def filter_interior_lines(
    interior_lines_map: Dict[str, List[LineString]],
    min_painting_width: float,
) -> List[LineString]:
    f_interior_lines_map = {}
    for wall_id, wall_lines in interior_lines_map.items():
        new_lines = filter_wall_lines(wall_lines, min_painting_width)
        if len(new_lines) > 0:
            f_interior_lines_map[wall_id] = new_lines
    return f_interior_lines_map


def subtract_polygon(line: LineString, polygon: Polygon) -> List[LineString]:
    diff = line - polygon
    return list(diff.geoms) if isinstance(diff, MultiLineString) else [diff]


ALLOW_DUPLICATE_PAINTINGS_IN_HOUSE: bool = False
"""Allow for the same painting to appear multiple times in the house."""

VISUALIZE: bool = False


def add_paintings(
    controller: Controller,
    house: Dict[str, Any],
    line_segments: List[Tuple[Tuple[int, int], Tuple[int, int]]],
) -> None:
    paintings = asset_database["Painting"]
    paintings_df = pd.DataFrame(
        [
            {
                "assetId": painting["assetId"],
                "width": painting["boundingBox"]["x"],
                "height": painting["boundingBox"]["y"],
                "depth": painting["boundingBox"]["z"],
            }
            for painting in paintings
        ]
    )
    min_painting_width = paintings_df["width"].min()

    interior_walls = [wall for wall in house["walls"] if wall["roomId"] is not None]

    interior_lines_map = {}
    walls = {wall["id"]: wall for wall in house["walls"]}

    for wall in interior_walls:
        p0 = wall["polygon"][0]
        p1 = wall["polygon"][1]
        line = LineString([(p0["x"], p0["z"]), (p1["x"], p1["z"])])
        interior_lines_map[wall["id"]] = [line]

    # NOTE: Subtract wall objects
    WALL_OBJECT_PADDING = 0.15
    for wall_obj in house["windows"] + house["doors"]:
        hole = wall_obj["holePolygon"]
        hole_start_delta = hole[0]["x"]
        hole_end_delta = hole[1]["x"]

        wall0_id = wall_obj["wall0"]
        wall1_id = wall_obj["wall1"]
        for wall in house["walls"]:
            walls.keys()

        p0 = walls[wall0_id]["polygon"][0]
        p1 = walls[wall0_id]["polygon"][1]

        direction = np.array([p1["x"] - p0["x"], p1["z"] - p0["z"]])
        direction /= np.linalg.norm(direction)
        hole_start = np.array([p0["x"], p0["z"]]) + direction * hole_start_delta
        hole_end = np.array([p0["x"], p0["z"]]) + direction * hole_end_delta

        # rotate direction 90 degrees
        ortho_direction = np.array([-direction[1], direction[0]])

        hole_poly = Polygon(
            [
                hole_start
                - direction * WALL_OBJECT_PADDING / 2
                - ortho_direction * 0.5,
                hole_start
                - direction * WALL_OBJECT_PADDING / 2
                + ortho_direction * 0.5,
                hole_end + direction * WALL_OBJECT_PADDING / 2 + ortho_direction * 0.5,
                hole_end + direction * WALL_OBJECT_PADDING / 2 - ortho_direction * 0.5,
            ]
        )

        for wall_id in (wall0_id, wall1_id):
            if wall_id in interior_lines_map:
                # move hole_start_delta from (x0, z0) in the direction of (x1, z1)
                new_lines = []
                for interior_line in interior_lines_map[wall_id]:
                    new_lines += subtract_polygon(interior_line, hole_poly)
                for line in new_lines.copy():
                    if line.length < min_painting_width:
                        new_lines.remove(line)
                if len(new_lines) > 0:
                    interior_lines_map[wall_id] = new_lines
                else:
                    del interior_lines_map[wall_id]

    # NOTE: Subtract floor objects
    event = controller.reset(scene=house, renderImage=False)
    for obj in event.metadata["objects"]:
        if (
            obj["objectId"].startswith("wall")
            or obj["objectId"].startswith("room")
            or obj["objectId"].startswith("Floor")
            or obj["objectId"].startswith("Wall")
            or obj["objectId"].startswith("small|")
        ):
            continue
        bbox = obj["axisAlignedBoundingBox"]
        min_x = bbox["center"]["x"] - bbox["size"]["x"] / 2
        min_y = bbox["center"]["y"] - bbox["size"]["y"] / 2
        min_z = bbox["center"]["z"] - bbox["size"]["z"] / 2
        max_x = bbox["center"]["x"] + bbox["size"]["x"] / 2
        max_y = bbox["center"]["y"] + bbox["size"]["y"] / 2
        max_z = bbox["center"]["z"] + bbox["size"]["z"] / 2

        if max_y > 1.25:
            obj_top_down_poly = Polygon(
                [
                    (min_x - 0.1, min_z - 0.1),
                    (min_x - 0.1, max_z + 0.1),
                    (max_x + 0.1, max_z + 0.1),
                    (max_x + 0.1, min_z - 0.1),
                ]
            )
            for wall_id, wall_lines in interior_lines_map.items():
                new_lines = []
                for wall_line in wall_lines:
                    new_lines += subtract_polygon(wall_line, obj_top_down_poly)
                interior_lines_map[wall_id] = new_lines

    if VISUALIZE:
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))

        ax = axs[0]
        for wall_id, lines in interior_lines_map.items():
            for line in lines:
                (x0, z0), (x1, z1) = line.coords
                ax.plot([x0, x1], [z0, z1], color="red", alpha=0.25)

        ax = axs[1]

    interior_lines_map = filter_interior_lines(interior_lines_map, min_painting_width)

    if VISUALIZE:
        for (x0, z0), (x1, z1) in line_segments:
            ax.plot([x0, x1], [z0, z1], color="black", alpha=0.25)
        ax.set_aspect("equal")

    wall_metadata = {
        obj["objectId"]: obj
        for obj in event.metadata["objects"]
        if obj["objectType"] == "Wall"
    }
    painting_count = 0
    for wall_id, wall_lines in interior_lines_map.items():
        paintings_on_wall_p = random.uniform(0.3, 0.7)
        print("paintings_on_wall_p:", paintings_on_wall_p)
        while wall_lines:
            # NOTE: Randomly skip some of the walls
            if random.random() < paintings_on_wall_p:
                break

            wall = walls[wall_id]
            wall_height = wall["polygon"][2]["y"]

            wall_line = random.choice(wall_lines)
            line_length = wall_line.length
            (x0, z0), (x1, z1) = wall_line.coords

            top_open_position = min(3, wall_height)
            bottom_open_position = random.choice([0, 0.5, 1.25])

            valid_paintings = paintings_df[
                (paintings_df["width"] + 0.05 <= line_length)
                & (
                    paintings_df["height"] + 0.05
                    < top_open_position - bottom_open_position
                )
            ]

            if len(valid_paintings) == 0:
                break
            painting = valid_paintings.sample()

            # remove painting from paintings_df
            if not ALLOW_DUPLICATE_PAINTINGS_IN_HOUSE:
                paintings_df.drop(painting.index, inplace=True)

            painting_width = painting["width"].values[0]
            painting_height = painting["height"].values[0]
            painting_depth = painting["depth"].values[0]

            start_painting_x = random.random() * (line_length - painting_width)
            painting_center_x = start_painting_x + painting_width / 2

            # sample random normal
            rand_range = top_open_position - bottom_open_position - painting_height
            assert rand_range >= 0
            painting_center_y = (
                bottom_open_position
                + painting_height / 2
                + rand_range * np.random.beta(a=2.5, b=2.5)
            )

            direction = np.array([x1 - x0, z1 - z0])
            direction /= np.linalg.norm(direction)
            angle = wall_metadata[wall_id]["rotation"]["y"] + 180

            painting_center = np.array([x0, z0]) + direction * painting_center_x
            if VISUALIZE:
                ax.scatter(painting_center[0], painting_center[1], color="red")

            # rotate direction 90 degrees
            ortho_direction = np.array([-direction[1], direction[0]])

            # move the paiting 5 meters in ortho_angle's direction
            painting_center -= ortho_direction * (painting_depth / 2 + 0.01)
            if VISUALIZE:
                ax.scatter(painting_center[0], painting_center[1], color="blue")

            house["objects"].append(
                {
                    "assetId": painting["assetId"].values[0],
                    "id": f"Painting|{painting_count}",
                    "rotation": {"x": 0, "y": angle, "z": 0},
                    "position": {
                        "x": painting_center[0],
                        "y": painting_center_y,
                        "z": painting_center[1],
                    },
                    "kinematic": True,
                }
            )
            painting_count += 1

            polygon_points = [
                [
                    *painting_center
                    + direction * painting_width / 2
                    + ortho_direction * (painting_depth + 0.1)
                ],
                [
                    *painting_center
                    + direction * painting_width / 2
                    - ortho_direction * (painting_depth + 0.1)
                ],
                [
                    *painting_center
                    - direction * painting_width / 2
                    - ortho_direction * (painting_depth + 0.1)
                ],
                [
                    *painting_center
                    - direction * painting_width / 2
                    + ortho_direction * (painting_depth + 0.1)
                ],
            ]

            painting_polygon = Polygon(polygon_points)

            # plot the polygon_points
            if VISUALIZE:
                for (x0, z0), (x1, z1) in zip(
                    polygon_points, polygon_points[1:] + [polygon_points[0]]
                ):
                    ax.plot([x0, x1], [z0, z1], color="green")

            new_wall_lines = []
            for line in wall_lines:
                new_wall_lines += subtract_polygon(line, painting_polygon)
            wall_lines = filter_wall_lines(new_wall_lines, min_painting_width)
