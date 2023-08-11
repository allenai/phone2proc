import random
from typing import Any, Dict

import numpy as np

from procthor.databases import asset_database

IOU_P_FILTER = 0.75
"""Only objects within this percentage of the best IOU will be considered for sampling."""


def bbox_iou(bbox1, bbox2) -> float:
    """3D bounding box intersection over union of bbox sizes."""
    intersection = np.prod(np.min(np.array([bbox1, bbox2]), axis=0))
    union = np.prod(bbox1) + np.prod(bbox2) - intersection
    return intersection / union


def rank_objects_by_iou(object_type: str, bbox) -> Dict[str, Any]:
    """Sorts the cloests THOR objects of the given type based on bbox iou."""
    if object_type == "Television":
        thor_object_types = ["Television"]
    elif object_type == "Table":
        # Table may also be a side table or dining table
        thor_object_types = ["CoffeeTable", "SideTable", "DiningTable"]
    elif object_type == "Chair":
        thor_object_types = ["Chair"]
    elif object_type == "Bed":
        thor_object_types = ["Bed"]
    elif object_type == "Storage":
        thor_object_types = ["ShelvingUnit", "CounterTop"]
    elif object_type == "Sofa":
        # Sofa may also be an arm chair
        thor_object_types = ["Sofa", "ArmChair"]
    elif object_type == "Refrigerator":
        thor_object_types = {"Fridge"}
    elif object_type == "Toilet":
        thor_object_types = {"Toilet"}
    elif object_type == "Fireplace":
        thor_object_types = {"SideTable", "CoffeeTable", "DiningTable", "CounterTop"}
    else:
        raise NotImplementedError(f"object_type {object_type} not implemented")

    out_objects = []
    for candidate_type in thor_object_types:
        for candidate_obj in asset_database[candidate_type]:
            candidate_obj_bbox = np.array(list(candidate_obj["boundingBox"].values()))
            iou = bbox_iou(bbox, candidate_obj_bbox)
            out_objects.append(dict(obj=candidate_obj, iou=iou))
    out_objects.sort(key=lambda x: x["iou"], reverse=True)
    return out_objects


def select_thor_objects(objects):
    for obj in objects:
        obj_rankings = rank_objects_by_iou(obj["object_type"], obj["bbox"])
        best_iou = obj_rankings[0]["iou"]
        obj_candidates = [
            o["obj"] for o in obj_rankings if o["iou"] > IOU_P_FILTER * best_iou
        ]
        obj["ai2thor_object"] = random.choice(obj_candidates)
