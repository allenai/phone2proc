import random
from typing import Any, Dict

import numpy as np
from shapely.geometry import LineString, Point

CHAIR_TABLE_DIST_THRESHOLD = 0.25
"""
If the nearest chair's axis is within MOVE_CHAIR_AXIS_MAX_DIST of the nearest table's bounding box line segment, then randomize the chair's pose.
"""

P_RANDOMIZE_CHAIR_ROTATION = 0.8

P_RANDOMIZE_CHAIR_POSITION = 0.8


def plot_chairs_and_tables(ax, objects, color: str = "red"):
    for obj in objects:
        if obj["object_type"] not in {"Chair", "Table"}:
            continue
        rect = obj["top_down_rect"].copy()
        rect.append(rect[0])
        ax.plot(*zip(*rect), color=color)
    ax.set_aspect("equal")
    return ax


def sample_chair_position_delta() -> float:
    """Sample the distance to move the chair."""
    return np.random.beta(a=1.40, b=5.2)


def sample_chair_rotation_delta() -> float:
    """Sample the angle to rotate the chair."""
    return np.random.normal(loc=0, scale=15)


def randomize_chair_poses(objects: Dict[str, Any]) -> None:
    # NOTE: Get the line segments that make up each table's bounding box
    # and each table's object oriented axes.
    tables_line_segments = {}
    table_axes = {}
    for obj in objects:
        if obj["object_type"] not in {"Table"}:
            continue

        rect = obj["top_down_rect"]
        rect.append(rect[0])

        center = np.mean(rect, axis=0)
        rot_deg = obj["room_plan_rotation"]
        axes = []
        for rot_delta in range(0, 360, 90):
            rot = (rot_deg + rot_delta) * np.pi / 180
            axis = [
                [center[0], center[1]],
                [
                    center[0] + np.cos(rot) * 1,
                    center[1] + np.sin(rot) * 1,
                ],
            ]
            axes.append(axis)
        table_axes[obj["file"]] = axes

        tables_line_segments[obj["file"]] = [
            (rect[i], rect[i + 1]) for i in range(len(rect) - 1)
        ]

    # NOTE: Find the nearest table for each chair.
    closest_chair_table_pairs = []
    for obj in objects:
        if obj["object_type"] != "Chair":
            continue

        mid_x, mid_z = obj["position"][0], obj["position"][2]

        # find the closest table line segment
        closest_table_line_segment = None
        closest_table_line_segment_dist = None
        for table_file, table_line_segments in tables_line_segments.items():
            for line_i, line_segment in enumerate(table_line_segments):
                line = LineString(line_segment)
                dist = line.distance(Point(mid_x, mid_z))
                if (
                    closest_table_line_segment is None
                    or dist < closest_table_line_segment_dist
                ):
                    closest_table_line_segment = table_file, line_i
                    closest_table_line_segment_dist = dist
        if closest_table_line_segment_dist > CHAIR_TABLE_DIST_THRESHOLD:
            continue
        closest_chair_table_pairs.append((obj["file"], *closest_table_line_segment))

    # NOTE: Find the closest line segment for each chair
    chair_axes = {}
    for chair_file, table_file, table_line_i in closest_chair_table_pairs:
        line_segment = tables_line_segments[table_file][table_line_i]
        line_segment_x = (line_segment[0][0] + line_segment[1][0]) / 2
        line_segment_z = (line_segment[0][1] + line_segment[1][1]) / 2

        nearest_axis = None
        nearest_axis_dist = float("inf")
        for axis_i, (point_1, (axis_p2_x, axis_p2_z)) in enumerate(
            table_axes[table_file]
        ):
            dist = (
                (line_segment_x - axis_p2_x) ** 2 + (line_segment_z - axis_p2_z) ** 2
            ) ** 0.5
            if dist < nearest_axis_dist:
                nearest_axis = table_axes[table_file][axis_i]
                nearest_axis_dist = dist
        chair_axes[chair_file] = nearest_axis

    # NOTE: Get the angle to push the chair back at
    chair_pushback_angles = {}
    for chair_file, (
        (axis_p0_x, axis_p0_z),
        (axis_p1_x, axis_p1_z),
    ) in chair_axes.items():
        # NOTE: get the angle between point0 and point1
        angle = (
            np.arctan2(axis_p1_z - axis_p0_z, axis_p1_x - axis_p0_x) * 180 / np.pi
        ) % 360
        chair_pushback_angles[chair_file] = angle

    # NOTE: Randomize the chair's pose
    for obj in objects:
        if obj["object_type"] != "Chair":
            continue
        if obj["file"] not in chair_pushback_angles:
            continue

        angle = chair_pushback_angles[obj["file"]]

        # NOTE: randomize the position
        if random.random() < P_RANDOMIZE_CHAIR_POSITION:
            # NOTE: push the chair back a bit
            move_back_length = sample_chair_position_delta()

            # NOTE: update top_down_rect
            for rect_i, (x, z) in enumerate(obj["top_down_rect"]):
                obj["top_down_rect"][rect_i] = (
                    x + move_back_length * np.cos(angle * np.pi / 180),
                    z + move_back_length * np.sin(angle * np.pi / 180),
                )

            # NOTE: update the position
            obj["position"][0] += move_back_length * np.cos(angle * np.pi / 180)
            obj["position"][2] += move_back_length * np.sin(angle * np.pi / 180)

        # NOTE: randomize the rotation
        if random.random() < P_RANDOMIZE_CHAIR_ROTATION:
            delta_rot = sample_chair_rotation_delta()
            for rect_i, (x, z) in enumerate(obj["top_down_rect"]):
                # NOTE: Rotate the top_down_rect in place
                obj["top_down_rect"][rect_i] = (
                    (x - obj["position"][0]) * np.cos(delta_rot * np.pi / 180)
                    - (z - obj["position"][2]) * np.sin(delta_rot * np.pi / 180)
                    + obj["position"][0],
                    (x - obj["position"][0]) * np.sin(delta_rot * np.pi / 180)
                    + (z - obj["position"][2]) * np.cos(delta_rot * np.pi / 180)
                    + obj["position"][2],
                )

            # NOTE: update the rotation
            obj["rotation"] += delta_rot

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plot_chairs_and_tables(ax, objects)
    fig.savefig("temp.png")
