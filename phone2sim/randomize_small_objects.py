#%%
"""
Add small objects to the house.

Also randomizes the openness of some of the objects.
"""

import copy
import logging
import random
from collections import defaultdict
from typing import Dict, List

import numpy as np
from ai2thor.controller import Controller
from procthor.constants import FLOOR_Y, OPENNESS_RANDOMIZATIONS
from procthor.databases import asset_id_database, assets_df
from procthor.databases import objects_in_receptacles as orig_objects_in_receptacles
from procthor.databases import placement_annotations
from procthor.generation.objects import sample_openness
from procthor.utils.types import Object, Split, Vector3

PARENT_BIAS = defaultdict(
    lambda: 0.2, {"Chair": 0, "ArmChair": 0, "Countertop": 0.2, "ShelvingUnit": 0.4}
)
"""The sampling bias for how often an object is placed in a receptacle."""

CHILD_BIAS = defaultdict(
    lambda: 0,
    {
        "HousePlant": 0.25,
        "Basketball": 0.2,
        "SprayBottle": 0.2,
        "Pot": 0.1,
        "Pan": 0.1,
        "Bowl": 0.05,
        "BaseballBat": 0.1,
    },
)
"""The sampling bias for how often an individual object is sampled."""


def randomize_bias():
    """Randomize the sampling bias for the parent and child objects."""
    lower = -0.3
    upper = 0.1
    return (upper - lower) * np.random.beta(a=3.5, b=1.9) + lower


MAX_OF_TYPE_ON_RECEPTACLE = 3
"""The maximum number of objects of a given type that can be placed on a receptacle."""

HOUSE_PLANT_MAX_HEIGHT = 0.75
"""The maximum height of a house plant that is considered small.

Note: There are some large house plants are intended to be placed on the floor.
"""

FLOOR_OBJECTS_TO_DROP = {"BaseballBat": {"p": 0.7}}
"""The objects that are dropped to be put in place."""

OBJECTS_TO_DROP = {"BaseballBat": {"p": 0.97}}
"""The objects that are pushed when they are placed on a receptacle.

Helps avoid the baseball bat from always standing upright.
"""


def add_small_objects(
    partial_house,
    controller: Controller,
) -> None:
    """Add small objects to the house."""
    # NOTE: add biases to the target objects so they're more likely to appear
    objects_in_receptacles = copy.deepcopy(orig_objects_in_receptacles)

    # NOTE: fridges won't be opening, so remove
    del objects_in_receptacles["Fridge"]

    # NOTE: delete all Tomatos, visually appear nearly identical to Apples
    for parent, children in objects_in_receptacles.items():
        if "Tomato" in children:
            del children["Tomato"]
        if "Bottle" in children:
            del children["Bottle"]

    for TARGET_OBJECT in {"Apple", "Vase"}:
        ranked_parents = []
        for parent, children in objects_in_receptacles.items():
            for child, child_meta in children.items():
                if child == TARGET_OBJECT:
                    ranked_parents.append((parent, child_meta))
        ranked_parents.sort(key=lambda x: x[1]["p"], reverse=True)

        # add between 0.3 and 0.1 to the bias to rank more probable parents higher
        for parent_i, (parent, child_meta) in enumerate(ranked_parents):
            child_meta["p"] += 0.2 * (1 - parent_i / (len(ranked_parents) - 1)) + 0.1
        ranked_parents

    event = controller.reset(scene=partial_house)
    assert event, "Unable to CreateHouse!"
    controller.step(action="SetObjectFilter", objectIds=[])

    # NOTE: Get the objects in the room.
    objects = [
        obj
        for obj in event.metadata["objects"]
        if not any(
            obj["objectId"].startswith(k)
            for k in ["wall|", "room|", "Floor", "door|", "window|"]
        )
    ]
    objects_per_room = defaultdict(list)
    for obj in objects:
        # NOTE: assumes only one room
        room_id = "room"
        objects_per_room["room"].append(obj)
    objects_per_room = dict(objects_per_room)
    receptacles_per_room = {
        room_id: [obj for obj in objects if obj["objectType"] in objects_in_receptacles]
        for room_id, objects in objects_per_room.items()
    }
    object_types_in_rooms = {
        room_id: set(obj["objectType"] for obj in objects)
        for room_id, objects in objects_per_room.items()
    }

    objects_in_house = {obj["id"]: obj for obj in partial_house["objects"]}

    house_bias = randomize_bias()
    logging.debug(f"Small object bias: {house_bias}")

    # NOTE: Place the objects
    placed_objects = 0
    for room_id in object_types_in_rooms:
        if room_id not in receptacles_per_room:
            continue
        receptacles_in_room = receptacles_per_room[room_id]
        # room_type = room.room_type
        spawnable_objects = []
        for receptacle in receptacles_in_room:
            objects_in_receptacle = objects_in_receptacles[receptacle["objectType"]]
            for object_type, data in objects_in_receptacle.items():
                # room_weight = placement_annotations.loc[object_type][f"in{room_type}s"]
                # if room_weight == 0:
                #     continue
                spawnable_objects.append(
                    {
                        "receptacleId": receptacle["objectId"],
                        "receptacleType": receptacle["objectType"],
                        "childObjectType": object_type,
                        "pSpawn": data["p"],
                    }
                )

        filtered_spawnable_groups = [
            group
            for group in spawnable_objects
            if random.random()
            <= (
                group["pSpawn"]
                + PARENT_BIAS[group["receptacleType"]]
                + CHILD_BIAS[group["childObjectType"]]
                + house_bias
            )
        ]
        random.shuffle(filtered_spawnable_groups)

        # NOTE: Apple and Vase are priority object types.
        # move Apple and Vased childObjectType to the front of the list
        if random.random() < 0.5:
            filtered_spawnable_groups.sort(
                key=lambda x: x["childObjectType"] in ["Apple", "Vase"], reverse=True
            )

        for group in filtered_spawnable_groups:
            # NOTE: Supports things like 3 plates on a surface.
            num_of_type = 1
            # NOTE: intentionally has no bias on > 1 samples.
            while random.random() <= group["pSpawn"]:
                num_of_type += 1
                if num_of_type >= MAX_OF_TYPE_ON_RECEPTACLE:
                    break

            for _ in range(num_of_type):
                # NOTE: Check if there can be multiple of the same type in the room.
                if (
                    group["childObjectType"] in object_types_in_rooms[room_id]
                    and not placement_annotations.loc[group["childObjectType"]][
                        "multiplePerRoom"
                    ]
                ):
                    break

                asset_candidates = assets_df[
                    (assets_df["objectType"] == group["childObjectType"])
                    # & assets_df["split"].isin([split, None])
                ]

                if group["childObjectType"] == "HousePlant":
                    # NOTE: House plants are a weird exception where there are massive
                    # house plants meant to be placed on the floor, and smaller
                    # house plants that can be placed on receptacles. This filters
                    # to only place smaller house plants on receptacles.
                    asset_candidates = asset_candidates[
                        asset_candidates["ySize"] < HOUSE_PLANT_MAX_HEIGHT
                    ]

                # NOTE: Some objects have multiple sim object receptacles within it,
                # so we need to specify all of them as possible receptacle object ids.
                event = controller.step(action="ResetObjectFilter")
                receptacle_object_ids = [
                    obj["objectId"]
                    for obj in event.metadata["objects"]
                    if obj["objectId"].startswith(group["receptacleId"])
                ]

                chosen_asset_id = asset_candidates.sample()["assetId"].iloc[0]

                generated_object_id = f"small|{room_id}|{placed_objects}"

                # NOTE: spawn below the floor so it doesn't tip over any other objects.
                event = controller.step(
                    action="SpawnAsset",
                    assetId=chosen_asset_id,
                    generatedId=generated_object_id,
                    position=Vector3(x=0, y=FLOOR_Y - 20, z=0),
                    renderImage=False,
                )
                assert (
                    event
                ), f"SpawnAsset failed for {chosen_asset_id} with {event.metadata['actionReturn']}!"
                controller.step(
                    action="SetObjectFilter", objectIds=[generated_object_id]
                )

                obj_type = asset_id_database[chosen_asset_id]["objectType"]
                openness = None
                if (
                    obj_type in OPENNESS_RANDOMIZATIONS
                    and "CanOpen"
                    in asset_id_database[chosen_asset_id]["secondaryProperties"]
                ):
                    openness = sample_openness(obj_type)
                    controller.step(
                        action="OpenObject",
                        objectId=generated_object_id,
                        openness=openness,
                        forceAction=True,
                        raise_for_failure=True,
                        renderImage=False,
                    )
                event = controller.step(
                    action="InitialRandomSpawn",
                    randomSeed=random.randint(0, 1_000_000_000),
                    objectIds=[generated_object_id],
                    receptacleObjectIds=receptacle_object_ids,
                    forceVisible=False,
                    allowFloor=False,
                    renderImage=False,
                    allowMoveable=True,
                )
                obj = next(
                    obj
                    for obj in event.metadata["objects"]
                    if obj["objectId"] == generated_object_id
                )
                center_position = obj["axisAlignedBoundingBox"]["center"].copy()

                # NOTE: Sometimes InitialRandomSpawn succeeds when it should
                # be failing. In these cases, the object will appear below
                # the floor.
                if event and center_position["y"] > FLOOR_Y:
                    if obj["breakable"]:
                        # NOTE: often avoids objects shattering upon initialization.
                        center_position["y"] += 0.05

                    states = {}
                    if openness is not None:
                        states["openness"] = openness

                    # NOTE: "___" is when there is a child SimObjPhysics of another
                    # SimObjPhysics object (e.g., drawers on dressers).
                    house_data_receptacle = group["receptacleId"]
                    if "___" in group["receptacleId"]:
                        house_data_receptacle = group["receptacleId"][
                            : group["receptacleId"].find("___")
                        ]
                    if "children" not in objects_in_house[house_data_receptacle]:
                        objects_in_house[house_data_receptacle]["children"] = []

                    objects_in_house[house_data_receptacle]["children"].append(
                        Object(
                            id=generated_object_id,
                            assetId=chosen_asset_id,
                            rotation=obj["rotation"],
                            position=center_position,
                            kinematic=bool(
                                placement_annotations.loc[group["childObjectType"]][
                                    "isKinematic"
                                ]
                            ),
                            **states,
                        )
                    )

                    placed_objects += 1
                    object_types_in_rooms[room_id].add(group["childObjectType"])
                else:
                    controller.step(
                        action="DisableObject",
                        objectId=generated_object_id,
                        renderImage=False,
                    )

    # NOTE: Drop object from near ceiling so it falls
    def _set_drop_heights(objects: List[Object], obj_types):
        for obj in objects:
            obj_type = asset_id_database[obj["assetId"]]["objectType"]
            if obj_type in obj_types and random.random() < obj_types[obj_type]["p"]:
                obj["position"]["y"] = 3
                obj["rotation"]["x"] = random.random() * 2 + 3
                obj["rotation"]["y"] = random.random() * 360
                changed_ids.add(obj["id"])
                logging.debug("Spawning BaseballBat")
            if "children" in obj:
                _set_drop_heights(obj["children"], obj_types=OBJECTS_TO_DROP)

    # NOTE: Get pose of dropped object
    def _save_new_heights(objects: List[Object]):
        for obj in objects:
            if obj["id"] in changed_ids:
                thor_obj = next(
                    o for o in event.metadata["objects"] if o["objectId"] == obj["id"]
                )
                obj["position"] = thor_obj["axisAlignedBoundingBox"]["center"].copy()
                obj["rotation"] = thor_obj["rotation"].copy()
            if "children" in obj:
                _save_new_heights(obj["children"])

    changed_ids = set()
    orig_objects = copy.deepcopy(partial_house["objects"])
    _set_drop_heights(objects=partial_house["objects"], obj_types=FLOOR_OBJECTS_TO_DROP)
    if changed_ids:
        controller.reset()
        event = controller.step(
            action="CreateHouse", house=partial_house, renderImage=False
        )
        assert event, "Unable to CreateHouse!"

        # NOTE: wait for objects to settle.
        last_objs = [
            obj for obj in event.metadata["objects"] if obj["objectId"] in changed_ids
        ]
        i = 0
        failed = False
        while True:
            i += 1
            if i > 1000:
                failed = True
                print("Objects not settling!")
                break

            event = controller.step(
                action="AdvancePhysicsStep",
                timeStep=0.01,
                allowAutoSimulation=True,
                renderImage=False,
            )
            objs = [
                obj
                for obj in event.metadata["objects"]
                if obj["objectId"] in changed_ids
            ]

            if all(
                all(
                    abs(obj["position"][k] - last_obj["position"][k]) < 1e-3
                    and abs(obj["rotation"][k] - last_obj["rotation"][k]) < 1e-3
                    for k in ["x", "y", "z"]
                )
                for obj, last_obj in zip(objs, last_objs)
            ):
                break
            last_objs = objs

        if failed:
            partial_house["objects"] = orig_objects
        else:
            _save_new_heights(objects=partial_house["objects"])
