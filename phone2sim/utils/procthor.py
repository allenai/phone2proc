from collections import defaultdict

import numpy as np


def get_starter_house():
    return {
        "proceduralParameters": {
            "ceilingMaterial": "CeramicTiles3",
            "lights": [],
            "skyboxId": "SkyMountain",
        },
        "walls": [],
        "doors": [],
        "windows": [],
        "metadata": {"schema": "1.0.0"},
    }


def add_objects_to_house(house, wall_height, objects):
    house["objects"] = []
    for obj in objects:
        rect_x_mean = np.mean([p[0] for p in obj["top_down_rect"]])
        rect_z_mean = np.mean([p[1] for p in obj["top_down_rect"]])
        house["objects"].append(
            {
                "assetId": obj["ai2thor_object"]["assetId"],
                "id": obj["file"].split("/")[-1].split(".")[0],
                "kinematic": True,
                "position": {
                    "x": rect_x_mean,
                    "y": obj["position"][1] + wall_height / 2,
                    "z": rect_z_mean,
                },
                "rotation": {
                    "x": 0,
                    "y": obj["rotation"],
                    "z": 0,
                },
            }
        )


def add_walls_and_floor_to_house(house, floor_y, ceiling_height, floor_polygons):
    house["rooms"] = []
    house["walls"] = []
    walls_with_order = defaultdict(list)
    for i, room_polygon in enumerate(floor_polygons):
        house["rooms"].append(
            {
                "ceilings": [],
                "children": [],
                "floorMaterial": {"name": "OrangeCabinet"},
                "floorPolygon": [
                    dict(x=point[0], y=0, z=point[1]) for point in room_polygon
                ],
                "id": f"room|{i}",
                "roomType": "Misc",
            }
        )
        for (x0, z0), (x1, z1) in zip(
            room_polygon, room_polygon[1:] + room_polygon[:1]
        ):
            wall_order = f"{min(x0, x1):.2f}|{min(z0, z1):.2f}|{max(x0, x1):.2f}|{max(z0, z1):.2f}"
            house["walls"].append(
                {
                    "id": f"wall|{i}|{wall_order}",
                    "material": {"name": "OrangeCabinet"},
                    "roomId": f"room|{i}",
                    "polygon": [
                        dict(x=x0, y=floor_y, z=z0),
                        dict(x=x1, y=floor_y, z=z1),
                        dict(x=x1, y=ceiling_height, z=z1),
                        dict(x=x0, y=ceiling_height, z=z0),
                    ],
                }
            )
            walls_with_order[wall_order].append(house["walls"][-1]["polygon"])

    for wall_order, walls in walls_with_order.items():
        if len(walls) == 1:
            # add exterior wall
            poly = list(reversed(walls[0]))
            house["walls"].append(
                {
                    "id": f"wall|exterior|{wall_order}",
                    "material": "OrangeCabinet",
                    "roomId": None,
                    "polygon": poly[2:] + poly[:2],
                }
            )
