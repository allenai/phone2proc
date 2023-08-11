import itertools
import random

import numpy as np
from ai2thor.controller import Controller
from shapely.geometry import Polygon


def add_agent_poses(house, controller: Controller) -> None:
    poly = Polygon()
    for room in house["rooms"]:
        room_poly = Polygon([[p["x"], p["z"]] for p in room["floorPolygon"]])
        poly = poly.union(room_poly)

    # subtract objects from the house poly
    controller.reset(scene=house)
    controller.step(action="CreateHouse", house=house)
    for obj in controller.last_event.metadata["objects"]:
        center = obj["axisAlignedBoundingBox"]["center"]
        size = obj["axisAlignedBoundingBox"]["center"]
        top_down_poly = Polygon(
            [
                [center["x"] - size["x"] / 2, center["z"] - size["z"] / 2],
                [center["x"] + size["x"] / 2, center["z"] - size["z"] / 2],
                [center["x"] + size["x"] / 2, center["z"] + size["z"] / 2],
                [center["x"] - size["x"] / 2, center["z"] + size["z"] / 2],
            ]
        )
        # subtract top_down_poly from poly
        poly = poly.difference(top_down_poly)

    # get poly bounds
    min_x, min_z, max_x, max_z = poly.bounds

    # create a grid over the poly bounds
    xs = np.arange(min_x, max_x, 0.25)
    zs = np.arange(min_z, max_z, 0.25)

    combinations = list(itertools.product(xs, zs))
    random.shuffle(combinations)

    for x, z in combinations:
        x_length = 0.5
        z_length = 0.5
        xz_poly = Polygon(
            [
                [x + dx * x_length / 2, z + dz * z_length / 2]
                for dx, dz in [[-1, -1], [1, -1], [1, 1], [-1, 1]]
            ]
        )
        if poly.contains(xz_poly):
            break
    else:
        raise Exception("No valid location found!")

    pos = dict(x=x, y=0.95, z=z)
    rot_y = random.randint(0, 3) * 90
    rot = dict(x=0, y=rot_y, z=0)

    if "metadata" not in house:
        house["metadata"] = {}
    house["metadata"]["agentPoses"] = {
        "arm": {"position": pos, "rotation": rot, "horizon": 30, "standing": True},
        "default": {"position": pos, "rotation": rot, "horizon": 30, "standing": True},
        "locobot": {
            "position": pos,
            "rotation": rot,
            "horizon": 30,
        },
        "stretch": {"position": pos, "rotation": rot, "horizon": 30, "standing": True},
    }
    for agent in house["metadata"]["agentPoses"]:
        event = controller.reset(agentMode=agent, scene=house)
        assert 0.9 < event.metadata["agent"]["position"]["y"] < 1
