import numpy as np
from shapely.geometry import Polygon

from procthor.databases import asset_id_database


def realign_scene(line_segments, objects, rotation_bias):
    # scenes often are often in non-90 degree rotations at the start, and may not
    # be centered. Here, we center it and try to get it at a 90 degree rotation
    # NOTE: center all line segments around the origin, then rotate them by the -rotation_bias
    rotation_bias_nrad = -rotation_bias * np.pi / 180
    x_max = max(max(x1, x2) for (x1, _), (x2, _) in line_segments)
    x_min = min(min(x1, x2) for (x1, _), (x2, _) in line_segments)
    z_max = max(max(z1, z2) for (_, z1), (_, z2) in line_segments)
    z_min = min(min(z1, z2) for (_, z1), (_, z2) in line_segments)
    x_mean = (x_max + x_min) / 2
    z_mean = (z_max + z_min) / 2
    for i, (p1, p2) in enumerate(line_segments):
        line_segments[i][0] = (p1[0] - x_mean, p1[1] - z_mean)
        line_segments[i][1] = (p2[0] - x_mean, p2[1] - z_mean)

        # rotate all line segments by the -rotation_bias
        line_segments[i][0] = (
            p1[0] * np.cos(rotation_bias_nrad) - p1[1] * np.sin(rotation_bias_nrad),
            p1[0] * np.sin(rotation_bias_nrad) + p1[1] * np.cos(rotation_bias_nrad),
        )
        line_segments[i][1] = (
            p2[0] * np.cos(rotation_bias_nrad) - p2[1] * np.sin(rotation_bias_nrad),
            p2[0] * np.sin(rotation_bias_nrad) + p2[1] * np.cos(rotation_bias_nrad),
        )

    for obj in objects:
        obj["rotation"] += rotation_bias
        object_rects = obj["top_down_rect"]
        for i, p in enumerate(object_rects):
            object_rects[i] = (p[0] - x_mean, p[1] - z_mean)
            object_rects[i] = (
                p[0] * np.cos(rotation_bias_nrad) - p[1] * np.sin(rotation_bias_nrad),
                p[0] * np.sin(rotation_bias_nrad) + p[1] * np.cos(rotation_bias_nrad),
            )
        if "Chair0" in obj["file"] or "Table1" in obj["file"]:
            print(obj["file"].split("/")[-1], obj["rotation"])

    return x_mean, z_mean


def snap_objects_to_floor(house, walls, objects):
    # NOTE: about to correct floor positions of objects
    # get the y position of the floor
    wall_floor_pos = None
    for wall in walls:
        new_wall_floor_pos = wall["pose"]["position"][1] - wall["pose"]["bbox"][1] / 2
        assert wall_floor_pos is None or abs(wall_floor_pos - new_wall_floor_pos) < 1e-3
        wall_floor_pos = new_wall_floor_pos

    # get which objects are on top of each other
    on_top_objects = []
    for obj in objects:
        # print(obj["file"], obj["position"][1] - obj["bbox"][1] / 2)
        obj_lowest_pos = obj["position"][1] - obj["bbox"][1] / 2
        if abs(obj_lowest_pos - wall_floor_pos) > 1e-3:
            # on top of another object
            on_top_objects.append(obj)

    # NOTE: This assumes only 1 layer of stacked objects
    on_top_object_files = set([obj["file"] for obj in on_top_objects])
    on_top_graph = {}
    for on_top_obj in on_top_objects:
        on_top_poly = Polygon(on_top_objects[0]["top_down_rect"])

        best_iou = 0
        best_floor_obj = None
        for obj in objects:
            if obj["file"] in on_top_object_files:
                continue
            obj_poly = Polygon(obj["top_down_rect"])

            # calculate the iou
            iou = (
                on_top_poly.intersection(obj_poly).area
                / on_top_poly.union(obj_poly).area
            )
            if iou > best_iou:
                best_iou = iou
                best_floor_obj = obj
        floor_obj_id = best_floor_obj["file"].split("/")[-1].split(".")[0]
        on_top_id = on_top_obj["file"].split("/")[-1].split(".")[0]
        on_top_graph[on_top_id] = floor_obj_id

    for obj in house["objects"]:
        bounding_box = asset_id_database[obj["assetId"]]["boundingBox"]
        # correct the position
        if obj["id"] in on_top_graph:
            bottom_obj = next(
                o for o in house["objects"] if o["id"] == on_top_graph[obj["id"]]
            )
            bottom_obj_bbox = asset_id_database[bottom_obj["assetId"]]["boundingBox"]
            obj["position"]["y"] = bottom_obj_bbox["y"] + bounding_box["y"] / 2
        else:
            obj["position"]["y"] = bounding_box["y"] / 2
    return on_top_graph
