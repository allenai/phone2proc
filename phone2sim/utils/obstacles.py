import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from ai2thor.controller import Controller
from rasterio.features import rasterize
from scipy.spatial import ConvexHull
from shapely.geometry import MultiPolygon, Polygon

from procthor.databases import asset_database

controller = Controller(branch="nanna")

RECT_TYPE = Tuple[int, Tuple[int, int, int, int]]

OBSTACLE_OBJECT_TYPES = [
    "BasketBall",
    "Boots",
    "Box",
    "Cart",
    "Chair",
    "DogBed",
    "Dumbbell",
    "FloorLamp",
    "Footstool",
    "GarbageBag",
    "GarbageCan",
    "HousePlant",
    "LaundryHamper",
    "Ottoman",
    "Pillow",
    "RoomDecor",
    "Safe",
    "Stool",
    "TeddyBear",
]


def get_room_polys(
    house: Dict[str, Any], thor_objects_metadata: List[Dict[str, Any]]
) -> Dict[str, Polygon]:
    object_polys = []
    for obj in thor_objects_metadata:
        if (
            obj["objectId"].startswith("wall")
            or obj["objectId"].startswith("room")
            or obj["objectId"].startswith("window")
            or obj["objectId"].startswith("Floor")
            or obj["objectId"].startswith("small|")
            or obj["objectType"] == "Painting"
        ):
            continue

        points = (
            obj["objectOrientedBoundingBox"]["cornerPoints"]
            if obj["objectOrientedBoundingBox"] is not None
            else obj["axisAlignedBoundingBox"]["cornerPoints"]
        )
        points = set([(round(p[0], 2), round(p[2], 2)) for p in points])
        if len(points) != 4:
            points = obj["axisAlignedBoundingBox"]["cornerPoints"]
            points = set([(round(p[0], 2), round(p[2], 2)) for p in points])
            assert len(points) == 4

        points = list(points)
        hull = ConvexHull(points)
        hull_points = [points[i] for i in hull.vertices]
        object_polys.append(hull_points)

    room_polys = {}
    for room in house["rooms"]:
        floor_poly = [(p["x"], p["z"]) for p in room["floorPolygon"]]
        poly = Polygon(floor_poly)
        for obj in object_polys:
            poly -= Polygon(obj).buffer(0.75)
        assert room["id"] not in room_polys, room["id"]
        room_polys[room["id"]] = poly

    return room_polys


def rasterize_poly(polygon: Union[Polygon, MultiPolygon]) -> np.ndarray:
    geoms = polygon.geoms if isinstance(polygon, MultiPolygon) else [polygon]
    min_x = float("inf")
    min_z = float("inf")
    max_x = float("-inf")
    max_z = float("-inf")
    for poly in geoms:
        min_x = min(min_x, min([p[0] for p in poly.exterior.coords]))
        min_z = min(min_z, min([p[1] for p in poly.exterior.coords]))
        max_x = max(max_x, max([p[0] for p in poly.exterior.coords]))
        max_z = max(max_z, max([p[1] for p in poly.exterior.coords]))

    scale = int(250 / max(max_x - min_x, max_z - min_z))
    translate_x = -1 * min_x * scale + 10
    translate_z = -1 * min_z * scale + 10

    scaled_polys = []
    for poly in geoms:
        scaled_poly = []
        for x, z in poly.exterior.coords:
            scaled_poly.append((x * scale + translate_x, (z * scale + translate_z)))
        scaled_polys.append(Polygon(scaled_poly))

    im = rasterize(scaled_polys, out_shape=(300, 300))
    mask = im.astype(bool)

    return mask


def get_max_area(mask: np.ndarray) -> Tuple[RECT_TYPE, List[RECT_TYPE]]:
    w = np.zeros(dtype=int, shape=mask.shape)
    h = np.zeros(dtype=int, shape=mask.shape)

    nrows, ncols = mask.shape
    skip = 0
    area_max = (0, (None, None, None, None))

    candidates = []

    for r in range(nrows):
        for c in range(ncols):
            if mask[r][c] == skip:
                continue
            if r == 0:
                h[r][c] = 1
            else:
                h[r][c] = h[r - 1][c] + 1
            if c == 0:
                w[r][c] = 1
            else:
                w[r][c] = w[r][c - 1] + 1
            minw = w[r][c]
            rc_area_max = (0, (None, None, None, None))
            for dh in range(h[r][c]):
                minw = min(minw, w[r - dh][c])
                area = (dh + 1) * minw
                if area > area_max[0]:
                    area_max = (area, (r - dh, c - minw + 1, r, c))
                if area > rc_area_max[0]:
                    rc_area_max = (area, (r - dh, c - minw + 1, r, c))
            candidates.append(rc_area_max)
    return area_max, candidates


def get_obstacle_objects() -> pd.DataFrame:
    df = []
    for obj_type in OBSTACLE_OBJECT_TYPES:
        objects = asset_database[obj_type]
        for obj in objects:
            if obj_type == "HousePlant" and obj["assetId"] not in {
                "Houseplant_11",
                "Houseplant_16",
                "Houseplant_18",
                "Houseplant_26",
                "Houseplant_27",
                "Houseplant_29",
                "Houseplant_6",
                "Houseplant_7",
                "Houseplant_17",
            }:
                continue
            df.append(
                {
                    "assetId": obj["assetId"],
                    "objectType": obj_type,
                    "xSize": obj["boundingBox"]["x"],
                    "ySize": obj["boundingBox"]["y"],
                    "zSize": obj["boundingBox"]["z"],
                }
            )
    df = pd.DataFrame(df)
    return df


def sample_obstacle_object(
    rect_x_size: float, rect_z_size: float, objects_df: pd.DataFrame
) -> Tuple[Optional[dict], Optional[dict]]:
    random.shuffle(OBSTACLE_OBJECT_TYPES)
    for obj_type in OBSTACLE_OBJECT_TYPES:
        obj_df = objects_df[
            (objects_df["objectType"] == obj_type)
            & (
                (
                    (objects_df["xSize"] < rect_x_size)
                    & (objects_df["zSize"] < rect_z_size)
                )
                | (
                    (objects_df["zSize"] < rect_x_size)
                    & (objects_df["xSize"] < rect_z_size)
                )
            )
        ]
        if len(obj_df) > 0:
            # sample from obj_df
            obj = obj_df.sample()
            obj_x_size = obj["xSize"].values[0]
            obj_z_size = obj["zSize"].values[0]

            place_normal = obj_x_size < rect_x_size and obj_z_size < rect_z_size
            place_rotated = obj_z_size < rect_x_size and obj_x_size < rect_z_size
            if place_normal and place_rotated:
                max_side_length = max(obj_x_size, obj_z_size)
                x_start = random.random() * (rect_x_size - max_side_length)
                z_start = random.random() * (rect_z_size - max_side_length)
                x_end = x_start + max_side_length
                z_end = z_start + max_side_length
                rotation = random.choice(range(360))
            else:
                if not place_rotated:
                    x_start = random.random() * (rect_x_size - obj_x_size)
                    x_end = x_start + obj_x_size
                    z_start = random.random() * (rect_z_size - obj_z_size)
                    z_end = z_start + obj_z_size
                    rotation = random.choice([0, 180])
                else:
                    x_start = random.random() * (rect_x_size - obj_z_size)
                    x_end = x_start + obj_z_size
                    z_start = random.random() * (rect_z_size - obj_x_size)
                    z_end = z_start + obj_x_size
                    rotation = random.choice([90, 270])

            asset_id = obj["assetId"].values[0]

            asset_placement = {
                "assetId": asset_id,
                "position": {
                    "x": (x_start + x_end) / 2,
                    "y": obj["ySize"].values[0] / 2,
                    "z": (z_start + z_end) / 2,
                },
                "rotation": {"x": 0, "y": rotation, "z": 0},
            }
            return asset_placement, {
                "xStart": x_start,
                "xEnd": x_end,
                "zStart": z_start,
                "zEnd": z_end,
            }
    return None, None


def add_obstacle_objects(
    house: Dict[str, Any], thor_objects_metadata: List[Dict[str, Any]]
) -> None:
    room_polys = get_room_polys(house, thor_objects_metadata)
    objects_df = get_obstacle_objects()

    obstacle_i = 0
    for _, room_poly in room_polys.items():
        room_bias = random.random() * 0.2 + 0.4
        while True:
            print("placing...")
            if room_poly.is_empty:
                print("room_poly is empty")
            if room_poly.is_empty or random.random() < room_bias:
                print("breaking!")
                print(f"room_bias: {room_bias}")
                break

            mask = rasterize_poly(room_poly)

            _, rect_candidates = get_max_area(mask)
            for _ in range(3):
                zs, xs = np.where(mask)
                mask_min_x = min(xs)
                mask_max_x = max(xs)
                mask_min_z = min(zs)
                mask_max_z = max(zs)

                # assert abs(mask_min_x - 10) <= 5 and abs(mask_min_z - 10) <= 5

                # choose the index
                i = random.randint(0, len(xs) - 1)
                x = xs[i]
                z = zs[i]

                # find the biggest rectangle that contains the point
                f_candidates = [
                    (area, (z1, x1, z2, x2))
                    for area, (z1, x1, z2, x2) in rect_candidates
                    if x1 <= x <= x2 and z1 <= z <= z2
                ]
                area, (z1, x1, z2, x2) = sorted(
                    f_candidates, key=lambda x: x[0], reverse=True
                )[0]

                room_poly_bounds = room_poly.bounds
                room_poly_x_length = room_poly_bounds[2] - room_poly_bounds[0]
                room_poly_z_length = room_poly_bounds[3] - room_poly_bounds[1]

                # reset the mask min
                x1 -= mask_min_x
                x2 -= mask_min_x
                z1 -= mask_min_z
                z2 -= mask_min_z
                mask_max_x -= mask_min_x
                mask_max_z -= mask_min_z
                mask_min_x = 0
                mask_min_z = 0

                x1_ = x1 / mask_max_x * room_poly_x_length + room_poly_bounds[0]
                x2_ = x2 / mask_max_x * room_poly_x_length + room_poly_bounds[0]
                z1_ = z1 / mask_max_z * room_poly_z_length + room_poly_bounds[1]
                z2_ = z2 / mask_max_z * room_poly_z_length + room_poly_bounds[1]

                x_size = x2_ - x1_
                z_size = z2_ - z1_

                asset_placement, top_down_rect = sample_obstacle_object(
                    x_size, z_size, objects_df
                )
                if asset_placement is None:
                    continue

                # remove sampled asset from objects_df
                objects_df = objects_df[
                    ~(objects_df["assetId"] == asset_placement["assetId"])
                ]

                top_down_rect["xStart"] += x1_
                top_down_rect["xEnd"] += x1_
                top_down_rect["zStart"] += z1_
                top_down_rect["zEnd"] += z1_

                asset_placement["position"]["x"] += x1_
                asset_placement["position"]["z"] += z1_

                asset_placement["id"] = f"obstacle_{obstacle_i}"
                asset_placement["kinematic"] = True
                house["objects"].append(asset_placement)
                obstacle_i += 1
                break
            else:
                break

            to_subtract = Polygon(
                [
                    (top_down_rect["xStart"], top_down_rect["zStart"]),
                    (top_down_rect["xStart"], top_down_rect["zEnd"]),
                    (top_down_rect["xEnd"], top_down_rect["zEnd"]),
                    (top_down_rect["xEnd"], top_down_rect["zStart"]),
                ]
            )
            room_poly -= to_subtract.buffer(0.5)
