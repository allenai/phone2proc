import copy
import math
import random
from itertools import combinations
from typing import Dict, List

import numpy as np
from ai2thor.controller import Controller
from phone2sim.utils.rank_objects import IOU_P_FILTER, rank_objects_by_iou
from shapely.geometry import LineString, Polygon
from trimesh import Trimesh
from trimesh.collision import CollisionManager

from procthor.databases import asset_id_database

MOVE_INTERSECTION_THRESHOLD = 0.3
MOVE_EPSILON = 0.025
MAX_REPLACE_OBJECTS_TO_TRY = 4


def get_obj_wall_intersections(obj, line_segments):
    """Get the line segments that intersect with the the object.

    Expects the object in house["objects] format.
    """
    bounding_box = asset_id_database[obj["assetId"]]["boundingBox"]
    top_down_rect = [
        [bounding_box["x"] / 2, bounding_box["z"] / 2],
        [bounding_box["x"] / 2, -bounding_box["z"] / 2],
        [-bounding_box["x"] / 2, -bounding_box["z"] / 2],
        [-bounding_box["x"] / 2, bounding_box["z"] / 2],
    ]
    # in radians
    rotation = obj["rotation"]["y"] * np.pi / 180

    # rotate the bounding box by rotation
    for i, (x, z) in enumerate(top_down_rect):
        top_down_rect[i][0] = x * math.cos(rotation) - z * math.sin(rotation)
        top_down_rect[i][1] = -x * math.sin(rotation) + z * math.cos(rotation)

    # translate the bounding box by the object's position
    for i, point in enumerate(top_down_rect):
        top_down_rect[i][0] += obj["position"]["x"]
        top_down_rect[i][1] += obj["position"]["z"]

    obj_poly = Polygon(top_down_rect)

    intersecting_lines = []
    for line in line_segments:
        # create a shapely line
        line_string = LineString([line[0], line[1]])

        # check if the line intersects the polygon
        if obj_poly.intersects(line_string):
            intersecting_lines.append([line, line_string])

    return obj_poly, intersecting_lines


def fix_wall_intersections(house, line_segments):
    # NOTE: Move objects such that they aren't overlapping with the wall
    # TODO: eventually also want to consider moving objects on top of each other together
    for obj in house["objects"]:
        poly, intersecting_lines = get_obj_wall_intersections(obj, line_segments)
        if not intersecting_lines:
            continue

        # for each intersecting line, get the amount of overlap
        for line in intersecting_lines:
            line_string = line[1]
            # get the amount of overlap
            overlap = poly.intersection(line_string)
            line.append(overlap.length)
            line.append(overlap)

        intersecting_lines.sort(key=lambda line: line[2], reverse=True)
        while any(
            poly.intersects(line_string) for _, line_string, _, _ in intersecting_lines
        ):
            for line in intersecting_lines:
                line_points, line_string, overlap_length, overlap_line_string = line
                if not poly.intersects(line_string):
                    continue

                # get the angle between the line_points
                line_angle = (
                    math.atan2(
                        line_points[1][1] - line_points[0][1],
                        line_points[1][0] - line_points[0][0],
                    )
                    * 180
                    / math.pi
                )
                orthogonal_candidate_1 = line_angle + 90
                orthogonal_candidate_2 = line_angle - 90

                mean_line_x = overlap_line_string.centroid.x
                mean_line_y = overlap_line_string.centroid.y

                vec_size = 0.2

                # create a new point vec_size away from (mean_line_x, mean_line_y)
                # at the angle orthogonal_candidate_1
                vec_1 = [
                    mean_line_x
                    + vec_size * math.cos(orthogonal_candidate_1 * math.pi / 180),
                    mean_line_y
                    + vec_size * math.sin(orthogonal_candidate_1 * math.pi / 180),
                ]
                vec_2 = [
                    mean_line_x
                    + vec_size * math.cos(orthogonal_candidate_2 * math.pi / 180),
                    mean_line_y
                    + vec_size * math.sin(orthogonal_candidate_2 * math.pi / 180),
                ]

                # choose the orthogonal candidate that is closest to the poly centroid
                poly_mean_x = poly.centroid.x
                poly_mean_y = poly.centroid.y
                dist_1 = math.sqrt(
                    (vec_1[0] - poly_mean_x) ** 2 + (vec_1[1] - poly_mean_y) ** 2
                )
                dist_2 = math.sqrt(
                    (vec_2[0] - poly_mean_x) ** 2 + (vec_2[1] - poly_mean_y) ** 2
                )
                orthogonal_candidate = (
                    orthogonal_candidate_1
                    if dist_1 < dist_2
                    else orthogonal_candidate_2
                )

                # move the polygon move_poly_dist in the direction of the orthogonal candidate
                move_poly_dist = 0.05
                while poly.intersects(line_string):
                    poly_points = []
                    for point in poly.exterior.coords:
                        x = point[0] + move_poly_dist * math.cos(
                            orthogonal_candidate * math.pi / 180
                        )
                        z = point[1] + move_poly_dist * math.sin(
                            orthogonal_candidate * math.pi / 180
                        )
                        poly_points.append((x, z))
                    poly = Polygon(poly_points)

        # update the obj
        obj["position"]["x"] = poly.centroid.x
        obj["position"]["z"] = poly.centroid.y


def bounding_boxes_intersect(
    bbox1: Dict[str, Dict[str, float]],
    bbox2: Dict[str, Dict[str, float]],
    epsilon: float = 1e-3,
) -> bool:
    """
    Bounding boxes should be in the form of (xyz "min" points, xyz "max points).
    """
    return (
        bbox1["x"]["min"] < bbox2["x"]["max"] + epsilon
        and bbox1["x"]["max"] > bbox2["x"]["min"] - epsilon
        and bbox1["y"]["min"] < bbox2["y"]["max"] + epsilon
        and bbox1["y"]["max"] > bbox2["y"]["min"] - epsilon
        and bbox1["z"]["min"] < bbox2["z"]["max"] + epsilon
        and bbox1["z"]["max"] > bbox2["z"]["min"] - epsilon
    )


def objects_intersect(object1_id: str, object2_id: str, controller: Controller) -> bool:
    """Checks if 2 object meshes are colliding."""
    # NOTE: assumes the objects are in the scene already
    collision_manager = CollisionManager()
    obj1_geo = controller.step(
        action="GetInSceneAssetGeometry",
        objectId=object1_id,
        triangles=True,
        renderImage=False,
        raise_for_failure=True,
    ).metadata["actionReturn"]
    obj1_geo[0].keys()
    obj2_geo = controller.step(
        action="GetInSceneAssetGeometry",
        objectId=object2_id,
        triangles=True,
        renderImage=False,
        raise_for_failure=True,
    ).metadata["actionReturn"]

    # print("start")
    for i, mesh_info in enumerate(obj1_geo + obj2_geo):
        # NOTE: Swaps y and z dimensions
        vertices = np.array([[p["x"], p["z"], p["y"]] for p in mesh_info["vertices"]])
        triangles = np.array(mesh_info["triangles"]).reshape(-1, 3)[:, [0, 2, 1]]
        collision_manager.add_object(
            name=str(i),
            mesh=Trimesh(vertices=vertices, faces=triangles),
        )
    # print("mid")
    is_colliding, colliding_meshes = collision_manager.in_collision_internal(
        return_names=True
    )
    # print("end")
    return is_colliding


def get_intersection_size(min1: float, max1: float, min2: float, max2: float) -> float:
    """Get the 1d intersection of 2 intervals."""
    return max(0, min(max1, max2) - max(min1, min2))


def get_intersecting_objects(thor_object_metadata, controller, on_top_graph) -> list:
    intersecting_objects = []
    for obj1, obj2 in combinations(thor_object_metadata, 2):
        if (
            obj1["objectId"] in on_top_graph
            and on_top_graph[obj1["objectId"]] == obj2["objectId"]
            or obj2["objectId"] in on_top_graph
            and on_top_graph[obj2["objectId"]] == obj1["objectId"]
        ):
            # NOTE: here, one object is on top of the other, ignore collisions.
            continue

        bbox1 = obj1["axisAlignedBoundingBox"]
        bbox1 = {
            k: {
                "min": bbox1["center"][k] - bbox1["size"][k] / 2,
                "max": bbox1["center"][k] + bbox1["size"][k] / 2,
            }
            for k in ["x", "y", "z"]
        }
        bbox2 = obj2["axisAlignedBoundingBox"]
        bbox2 = {
            k: {
                "min": bbox2["center"][k] - bbox2["size"][k] / 2,
                "max": bbox2["center"][k] + bbox2["size"][k] / 2,
            }
            for k in ["x", "y", "z"]
        }

        if bounding_boxes_intersect(bbox1, bbox2) and objects_intersect(
            obj1["objectId"], obj2["objectId"], controller
        ):
            intersecting_objects.append([obj1, obj2])
    return intersecting_objects


def get_thor_object_metadata(controller: Controller, house) -> List[Dict]:
    controller.reset()
    event = controller.step(action="CreateHouse", house=house)
    house_object_ids = set()
    for obj in house["objects"]:
        if "children" in obj:
            for child_obj in obj["children"]:
                house_object_ids.add(child_obj["id"])
        house_object_ids.add(obj["id"])
    return [
        obj for obj in event.metadata["objects"] if obj["objectId"] in house_object_ids
    ]


def fix_object_intersection_with_move(
    thor_obj1_meta, thor_obj2_meta, house, controller, on_top_graph, line_segments
) -> bool:
    """Fix object intersection by moving the objects away from the other.

    Only will move objects if they're intersection is relatively small in at least
    1 direction.
    """
    bbox1 = thor_obj1_meta["axisAlignedBoundingBox"]
    x1_min = bbox1["center"]["x"] - bbox1["size"]["x"] / 2
    x1_max = bbox1["center"]["x"] + bbox1["size"]["x"] / 2
    z1_min = bbox1["center"]["z"] - bbox1["size"]["z"] / 2
    z1_max = bbox1["center"]["z"] + bbox1["size"]["z"] / 2

    bbox2 = thor_obj2_meta["axisAlignedBoundingBox"]
    x2_min = bbox2["center"]["x"] - bbox2["size"]["x"] / 2
    x2_max = bbox2["center"]["x"] + bbox2["size"]["x"] / 2
    z2_min = bbox2["center"]["z"] - bbox2["size"]["z"] / 2
    z2_max = bbox2["center"]["z"] + bbox2["size"]["z"] / 2

    x_intersection = get_intersection_size(x1_min, x1_max, x2_min, x2_max)
    z_intersection = get_intersection_size(z1_min, z1_max, z2_min, z2_max)

    x_int_amount = x_intersection / min(bbox1["size"]["x"], bbox2["size"]["x"])
    z_int_amount = z_intersection / min(bbox1["size"]["z"], bbox2["size"]["z"])

    ho1_i, house_obj1 = next(
        (i, obj)
        for i, obj in enumerate(house["objects"])
        if obj["id"] == thor_obj1_meta["objectId"]
    )
    ho2_i, house_obj2 = next(
        (i, obj)
        for i, obj in enumerate(house["objects"])
        if obj["id"] == thor_obj2_meta["objectId"]
    )
    house_obj1_copy = copy.deepcopy(house_obj1)
    house_obj2_copy = copy.deepcopy(house_obj2)

    axis = None
    if x_int_amount <= z_int_amount and x_int_amount < MOVE_INTERSECTION_THRESHOLD:
        # NOTE: move the objects in the x direction
        axis = "x"
    elif z_int_amount <= x_int_amount and z_int_amount < MOVE_INTERSECTION_THRESHOLD:
        # NOTE: use z
        axis = "z"

    thor_object_metadata = get_thor_object_metadata(controller, house)
    old_intersecting_objects = get_intersecting_objects(
        thor_object_metadata, controller, on_top_graph
    )

    # NOTE: try moving the object first
    if axis is not None:
        intersection_size = x_intersection if axis == "x" else z_intersection
        for d1, d2 in [[0.5, 0.5], [0, 1], [1, 0]]:
            if house_obj1["position"][axis] > house_obj2["position"][axis]:
                house_obj1["position"][axis] = (
                    house_obj1_copy["position"][axis]
                    + (intersection_size + MOVE_EPSILON) * d1
                )
                house_obj2["position"][axis] = (
                    house_obj2_copy["position"][axis]
                    - (intersection_size + MOVE_EPSILON) * d2
                )
            else:
                house_obj1["position"][axis] = (
                    house_obj1_copy["position"][axis]
                    - (intersection_size + MOVE_EPSILON) * d1
                )
                house_obj2["position"][axis] = (
                    house_obj2_copy["position"][axis]
                    + (intersection_size + MOVE_EPSILON) * d2
                )

            thor_object_metadata = get_thor_object_metadata(controller, house)
            new_intersecting_objects = get_intersecting_objects(
                thor_object_metadata, controller, on_top_graph
            )

            _, wall_intersections1 = get_obj_wall_intersections(
                house_obj1, line_segments
            )
            _, wall_intersections2 = get_obj_wall_intersections(
                house_obj2, line_segments
            )

            # corrected the collisions!
            if (
                len(new_intersecting_objects) < len(old_intersecting_objects)
                and len(wall_intersections1) == 0
                and len(wall_intersections2) == 0
            ):
                return True
        else:
            house["objects"][ho1_i] = house_obj1_copy
            house["objects"][ho2_i] = house_obj2_copy
    else:
        # NOTE: try moving the objects by max_move_dist to avoid intersections
        # max_move_dist = 0.05  # TODO: parameterize this!
        for max_move_dist in [0.05, 0.1]:
            for dx1, dx2, dz1, dz2 in [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, -1],
            ]:
                house_obj1["position"]["x"] = (
                    house_obj1_copy["position"]["x"]
                    + (max_move_dist + MOVE_EPSILON) * dx1
                )
                house_obj2["position"]["x"] = (
                    house_obj2_copy["position"]["x"]
                    + (max_move_dist + MOVE_EPSILON) * dx2
                )
                house_obj1["position"]["z"] = (
                    house_obj1_copy["position"]["z"]
                    + (max_move_dist + MOVE_EPSILON) * dz1
                )
                house_obj2["position"]["z"] = (
                    house_obj2_copy["position"]["z"]
                    + (max_move_dist + MOVE_EPSILON) * dz2
                )

                thor_object_metadata = get_thor_object_metadata(controller, house)
                new_intersecting_objects = get_intersecting_objects(
                    thor_object_metadata, controller, on_top_graph
                )

                _, wall_intersections1 = get_obj_wall_intersections(
                    house_obj1, line_segments
                )
                _, wall_intersections2 = get_obj_wall_intersections(
                    house_obj2, line_segments
                )

                # corrected the collisions!
                if (
                    len(new_intersecting_objects) < len(old_intersecting_objects)
                    and len(wall_intersections1) == 0
                    and len(wall_intersections2) == 0
                ):
                    return True

    return False


def fix_object_intersections_with_removal(controller, house, on_top_graph) -> None:
    """Remove an object from the ones that have collisions."""
    # TODO: I think this should also double check wall intersections
    while True:
        thor_object_metadata = get_thor_object_metadata(controller, house)
        intersecting_objects = get_intersecting_objects(
            thor_object_metadata, controller, on_top_graph
        )
        if len(intersecting_objects) == 0:
            break
        print(f"have {len(intersecting_objects)} intersecting objects still!")
        removed_ids = set()
        for obj1, obj2 in intersecting_objects:
            if obj1["objectId"] in removed_ids or obj2["objectId"] in removed_ids:
                # NOTE: already removed an object here
                continue

            if obj1["objectType"] == "Chair" and obj2["objectType"] in {
                "Table",
                "CoffeeTable",
                "SideTable",
            }:
                # NOTE: always remove chairs
                remove_obj = obj1
            elif (
                obj1["objectType"] in {"Table", "CoffeeTable", "SideTable"}
                and obj2["objectType"] == "Chair"
            ):
                remove_obj = obj2
            else:
                # remove objects weighted by volume (larger objects should appear more often)
                volume1 = np.prod(list(obj1["axisAlignedBoundingBox"]["size"].values()))
                volume2 = np.prod(list(obj2["axisAlignedBoundingBox"]["size"].values()))
                remove_obj = (
                    obj2 if random.random() * (volume1 + volume2) < volume1 else obj1
                )

            obj_i = next(
                i
                for i, obj in enumerate(house["objects"])
                if obj["id"] == remove_obj["objectId"]
            )
            del house["objects"][obj_i]
            removed_ids.add(remove_obj["objectId"])


def sbbox(bounding_box):
    """Serializes a bounding box."""
    return tuple(round(v, 2) for v in list(bounding_box.values()))


def fix_object_intersection_with_replace(
    obj1, obj2, house, controller, objects, on_top_graph, line_segments
):
    print("trying replace", obj1["objectId"], obj2["objectId"])
    # Try fixing intersections by replacing them with different assets.
    if (
        obj1["objectId"] in on_top_graph.values()
        or obj1["objectId"] in on_top_graph.keys()
        or obj2["objectId"] in on_top_graph.values()
        or obj2["objectId"] in on_top_graph.keys()
    ):
        # TODO: should eventually support replacing on-top objects, but just focusing
        # on floor objects for now.
        raise Exception("Cannot replace on-top objects.")

    obj1_scan_repr = next(obj for obj in objects if obj["mesh_id"] == obj1["objectId"])
    obj2_scan_repr = next(obj for obj in objects if obj["mesh_id"] == obj2["objectId"])

    # NOTE: avoids just sampling objects that are solely different based on material
    obj1_chosen_object_bboxes = set()
    obj2_chosen_object_bboxes = set()

    obj1_curr_bbox = sbbox(asset_id_database[obj1["assetId"]]["boundingBox"])
    obj2_curr_bbox = sbbox(asset_id_database[obj2["assetId"]]["boundingBox"])
    obj1_chosen_object_bboxes.add(obj1_curr_bbox)
    obj2_chosen_object_bboxes.add(obj2_curr_bbox)

    objs_to_try = [[obj1], [obj2]]
    for i, (obj, prev_obj_bboxes) in enumerate(
        [
            (obj1_scan_repr, obj1_chosen_object_bboxes),
            (obj2_scan_repr, obj2_chosen_object_bboxes),
        ]
    ):
        obj_rankings = rank_objects_by_iou(obj["object_type"], obj["bbox"])
        best_iou = obj_rankings[0]["iou"]
        obj_candidates = [
            o["obj"] for o in obj_rankings if o["iou"] > IOU_P_FILTER * best_iou
        ]
        random.shuffle(obj_candidates)
        for obj_candidate in obj_candidates:
            obj_candidate_bbox = sbbox(obj_candidate["boundingBox"])
            if obj_candidate_bbox in prev_obj_bboxes:
                continue
            prev_obj_bboxes.add(obj_candidate_bbox)
            objs_to_try[i].append(obj_candidate)
            if len(objs_to_try[i]) >= MAX_REPLACE_OBJECTS_TO_TRY:
                break

    thor_object_metadata = get_thor_object_metadata(controller, house)
    orig_intersecting_objects = get_intersecting_objects(
        thor_object_metadata, controller, on_top_graph
    )

    oho_i1, orig_house_object1 = next(
        (i, obj)
        for i, obj in enumerate(house["objects"])
        if obj["id"] == obj1["objectId"]
    )
    oho_i2, orig_house_object2 = next(
        (i, obj)
        for i, obj in enumerate(house["objects"])
        if obj["id"] == obj2["objectId"]
    )
    orig_house_object1_copy = copy.deepcopy(orig_house_object1)
    for obj_to_try in objs_to_try[0][1:]:
        # try replacing the object
        orig_house_object1["assetId"] = obj_to_try["assetId"]
        orig_house_object1["position"]["y"] = obj_to_try["boundingBox"]["y"] / 2

        thor_object_metadata = get_thor_object_metadata(controller, house)
        new_intersecting_objects = get_intersecting_objects(
            thor_object_metadata, controller, on_top_graph
        )

        _, wall_intersections1 = get_obj_wall_intersections(
            orig_house_object1, line_segments
        )
        _, wall_intersections2 = get_obj_wall_intersections(
            orig_house_object2, line_segments
        )
        if (
            len(new_intersecting_objects) < len(orig_intersecting_objects)
            and len(wall_intersections1) == 0
            and len(wall_intersections2) == 0
        ):
            return True
        thor_obj1 = next(
            o for o in thor_object_metadata if o["objectId"] == obj1["objectId"]
        )
        thor_obj2 = next(
            o for o in thor_object_metadata if o["objectId"] == obj2["objectId"]
        )
        # if fix_object_intersection_with_move(
        #     thor_obj1, thor_obj2, house, controller, on_top_graph, line_segments
        # ):
        #     return True

    # NOTE: revert the replacements because it didn't work!
    house["objects"][oho_i1] = orig_house_object1_copy
    orig_house_object1 = orig_house_object1_copy

    orig_house_object2_copy = copy.deepcopy(orig_house_object2)
    for obj_to_try in objs_to_try[1][1:]:
        # try replacing the object
        orig_house_object2["assetId"] = obj_to_try["assetId"]
        orig_house_object2["position"]["y"] = obj_to_try["boundingBox"]["y"] / 2

        thor_object_metadata = get_thor_object_metadata(controller, house)
        new_intersecting_objects = get_intersecting_objects(
            thor_object_metadata, controller, on_top_graph
        )

        _, wall_intersections1 = get_obj_wall_intersections(
            orig_house_object1, line_segments
        )
        _, wall_intersections2 = get_obj_wall_intersections(
            orig_house_object2, line_segments
        )
        if (
            len(new_intersecting_objects) < len(orig_intersecting_objects)
            and len(wall_intersections1) == 0
            and len(wall_intersections2) == 0
        ):
            return True
        thor_obj1 = next(
            o for o in thor_object_metadata if o["objectId"] == obj1["objectId"]
        )
        thor_obj2 = next(
            o for o in thor_object_metadata if o["objectId"] == obj2["objectId"]
        )
        # if fix_object_intersection_with_move(
        #     thor_obj1, thor_obj2, house, controller, on_top_graph, line_segments
        # ):
        #     return True

    # NOTE: revert the replacements because it didn't work!
    house["objects"][oho_i2] = orig_house_object2_copy
    orig_house_object2 = orig_house_object2_copy

    return False
