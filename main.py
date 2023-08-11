#%%
import copy
import datetime
import filecmp
import json
import os
import random
import shutil
import time
import uuid
from collections import defaultdict
from zipfile import ZipFile

import numpy as np
import trimesh
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from PIL import Image
from shapely.geometry import Point, Polygon
from trimesh import Trimesh

from phone2sim import visualize
from phone2sim.add_lights import add_lights
from phone2sim.randomize_materials import randomize_wall_and_floor_materials
from phone2sim.randomize_skybox import add_skybox
from phone2sim.randomize_small_objects import add_small_objects
from phone2sim.utils.agent_poses import add_agent_poses
from phone2sim.utils.intersections import (
    fix_object_intersection_with_move,
    fix_object_intersection_with_replace,
    fix_object_intersections_with_removal,
    fix_wall_intersections,
    get_intersecting_objects,
    get_thor_object_metadata,
)
from phone2sim.utils.join_lines import (
    connect_line_segments,
    process_line_segments,
    separate_free_walls,
)
from phone2sim.utils.layer import assign_layer_to_rooms
from phone2sim.utils.objects import get_object_yaw, get_objects
from phone2sim.utils.obstacles import add_obstacle_objects
from phone2sim.utils.paintings import add_paintings
from phone2sim.utils.parsing import (
    get_line_segments_and_walls,
    get_object_pose,
    get_wall_and_object_files,
)
from phone2sim.utils.polygons import get_inner_polygons, l2_dist
from phone2sim.utils.preprocess import get_scene_usda
from phone2sim.utils.procthor import (
    add_objects_to_house,
    add_walls_and_floor_to_house,
    get_starter_house,
)
from phone2sim.utils.randomize_chair_poses import randomize_chair_poses
from phone2sim.utils.rank_objects import select_thor_objects
from phone2sim.utils.realignment import realign_scene, snap_objects_to_floor
from phone2sim.utils.windows import add_windows, get_wall_holes
from phone2sim.visualize import visualize_line_segments


def flip_x_axis(line_segments, objects) -> None:
    for i, line in enumerate(line_segments):
        line_segments[i] = [
            (-line[0][0], line[0][1]),
            (-line[1][0], line[1][1]),
        ]
    for obj in objects:
        obj["top_down_rect"] = [
            (-obj["top_down_rect"][0][0], obj["top_down_rect"][0][1]),
            (-obj["top_down_rect"][1][0], obj["top_down_rect"][1][1]),
            (-obj["top_down_rect"][2][0], obj["top_down_rect"][2][1]),
            (-obj["top_down_rect"][3][0], obj["top_down_rect"][3][1]),
        ]
        obj["position"][0] = -obj["position"][0]

        # reflect the rotation degrees about the y axis
        obj["rotation"] = -obj["rotation"]


# random.seed(44)
# np.random.seed(44)

# file_id = "robothor-room"
# scene_usdz = "r2-room.usdz"
scene_usdz = "kianas.usdz"

# get the scene usda
scene_usda_path = get_scene_usda(scene_usdz)

# This isn't completely necessary, and can be left empty by default,
# but sometimes meshes are placed by mistake by Apple, so we just skip them.
skip_mesh_ids_dict = defaultdict(
    set,
    {
        "robothor-room.usdz": {"Chair0", "Chair2"},
        "anis-office-2nd-go.usdz": {},
        # "robothor-go-3.usdz": {"Wall1", "Wall3", "Wall2", "Wall8"},
        "r2-room.usdz": {"Wall7", "Wall8"},
        "kianas.usdz": {
            "Stove0",
            "Oven0",
            "Storage8",
            "Storage6",
            "Storage10",
            "Storage1",
            "Storage2",
            "Dishwasher0",
            "Sink0",
        },
        "back-robothor-room.usdz": {
            "Dishwasher0",
            "Oven0",
            "Bathtub0",
            "Sink0",
            "Stove0",
            "Sink1",
        },
        "matts-apartment.usdz": [
            # "Wall6",
            # "Wall7",
            # "Wall8",
            "Stove0",
            "Sink0",
            "Sink1",
            "Oven0",
            "Bathtub0",
            "Dishwasher0",
        ],
    },
)
skip_mesh_ids = skip_mesh_ids_dict[scene_usdz]


def generate_house(controller: Controller):
    wall_and_object_files = get_wall_and_object_files(
        room_usda=scene_usda_path, skip_mesh_ids=skip_mesh_ids
    )
    wall_files = wall_and_object_files["walls"]
    object_files = wall_and_object_files["objects"]

    line_segments, walls = get_line_segments_and_walls(wall_files)

    # visualize_line_segments(line_segments, None)
    # wall = walls[7]
    # wall_mesh = Trimesh(
    #     vertices=wall["pose"]["points"],
    #     faces=np.array(wall["pose"]["faces"]).reshape(-1, 3),
    #     process=False,
    # )
    # axis = trimesh.creation.axis(origin_color=[1.0, 0, 0])
    # scene = trimesh.Scene([wall_mesh, axis])
    # scene.show()
    # wall_holes[walls[2]["file"]]

    wall_height = walls[0]["pose"]["bbox"][1]
    ceiling_height = wall_height
    floor_y = 0

    graph, rotation_bias = process_line_segments(line_segments)
    ideal_wall_rotations = [get_object_yaw(wall["pose"]["rotation"]) for wall in walls]
    connect_line_segments(line_segments, graph, ideal_wall_rotations)
    objects = get_objects(object_files)

    randomize_chair_poses(objects)

    select_thor_objects(objects)
    realign_scene(line_segments, objects, rotation_bias)
    flip_x_axis(line_segments, objects)

    # draw_scene(line_segments, objects)
    # visualize_line_segments(line_segments, None)
    full_line_segments = copy.deepcopy(line_segments)

    full_graph = copy.deepcopy(graph)
    free_wall_segments = separate_free_walls(line_segments, graph)

    house = get_starter_house()
    add_objects_to_house(house, wall_height, objects)

    floor_polygons, polygon_graphs = get_inner_polygons(line_segments, graph)
    add_walls_and_floor_to_house(
        house=house,
        floor_y=floor_y,
        ceiling_height=ceiling_height,
        floor_polygons=floor_polygons,
    )

    # NOTE: Add free wall segments to the house
    wall_file_map = {}
    shapely_polys = [Polygon(floor_polygon) for floor_polygon in floor_polygons]
    for wall_file_idx, (p0, p1) in free_wall_segments.items():
        x0, z0 = p0
        x1, z1 = p1
        wall_order = (
            f"{min(x0, x1):.2f}|{min(z0, z1):.2f}|{max(x0, x1):.2f}|{max(z0, z1):.2f}"
        )
        mid_point = (x0 + x1) / 2, (z0 + z1) / 2
        # find which polygon the wall is in
        room_id = None
        for i, poly in enumerate(shapely_polys):
            if poly.contains(Point(mid_point)):
                room_id = str(i)
                break
        room_id_str = room_id if room_id is not None else "exterior"
        wall_1 = {
            "id": f"wall|{room_id_str}|{wall_order}|1",
            "roomId": f"room|{room_id}",
            "material": "OrangeCabinet",
            "polygon": [
                dict(x=x0, y=floor_y, z=z0),
                dict(x=x1, y=floor_y, z=z1),
                dict(x=x1, y=floor_y + ceiling_height, z=z1),
                dict(x=x0, y=floor_y + ceiling_height, z=z0),
            ],
        }
        wall_2 = {
            "id": f"wall|{room_id_str}|{wall_order}|2",
            "roomId": f"room|{room_id}",
            "material": "OrangeCabinet",
            "polygon": [
                dict(x=x1, y=floor_y, z=z1),
                dict(x=x0, y=floor_y, z=z0),
                dict(x=x0, y=floor_y + ceiling_height, z=z0),
                dict(x=x1, y=floor_y + ceiling_height, z=z1),
            ],
        }
        house["walls"].append(wall_1)
        house["walls"].append(wall_2)
        line_p0 = full_line_segments[wall_file_idx][0]
        wall_1_dist = l2_dist((x0, z0), line_p0)
        wall_2_dist = l2_dist((x1, z1), line_p0)
        assert (wall_1_dist < 1e-4) ^ (wall_2_dist < 1e-4)
        wall_file_map[wall_files[wall_file_idx]] = (
            [wall_1, wall_2] if wall_1_dist < 1e-4 else ([wall_2, wall_1])
        )
    for wall_i in sorted(free_wall_segments, reverse=True):
        wall_files.pop(wall_i)

    randomize_wall_and_floor_materials(house)
    add_lights(house, ceiling_height, 2)
    add_skybox(house)

    # NOTE: Handle objects
    on_top_graph = snap_objects_to_floor(house, walls, objects)
    fix_wall_intersections(house, full_line_segments)

    # NOTE: fix object collisions
    thor_object_metadata = get_thor_object_metadata(controller, house)
    intersecting_objects = get_intersecting_objects(
        thor_object_metadata, controller, on_top_graph
    )
    for obj1, obj2 in intersecting_objects:
        successful_move = fix_object_intersection_with_move(
            obj1, obj2, house, controller, on_top_graph, full_line_segments
        )
        if successful_move:
            continue
        successful_replace = fix_object_intersection_with_replace(
            obj1, obj2, house, controller, objects, on_top_graph, full_line_segments
        )
    # # NOTE: last resort
    fix_object_intersections_with_removal(controller, house, on_top_graph)

    add_small_objects(house, controller)

    # NOTE: Add windows
    wall_holes = {wall["file"]: get_wall_holes(wall["pose"]) for wall in walls}
    # visualize_line_segments(line_segments, house)

    walls_by_order = defaultdict(list)
    for wall in house["walls"]:
        wall_order = "|".join(wall["id"].split("|")[2:])
        walls_by_order[wall_order].append(wall)

    for room_i in range(len(polygon_graphs)):
        polys_in_room = list(range(len(polygon_graphs[room_i])))
        for poly_start, poly_end in zip(polys_in_room, polys_in_room[1:] + [0]):
            wall_file_idx = polygon_graphs[room_i][poly_end]
            wall_file = wall_files[wall_file_idx]

            if wall_file in wall_file_map:
                continue

            x0, z0 = floor_polygons[room_i][poly_start]
            x1, z1 = floor_polygons[room_i][poly_end]
            wall_order = f"{min(x0, x1):.2f}|{min(z0, z1):.2f}|{max(x0, x1):.2f}|{max(z0, z1):.2f}"

            wall_1, wall_2 = walls_by_order[wall_order]

            start_wall_pos = line_segments[wall_file_idx][0]
            wall_1_p0 = (wall_1["polygon"][0]["x"], wall_1["polygon"][0]["z"])
            wall_2_p0 = (wall_2["polygon"][0]["x"], wall_2["polygon"][0]["z"])
            wall_1_p0_dist = l2_dist(start_wall_pos, wall_1_p0)
            wall_2_p0_dist = l2_dist(start_wall_pos, wall_2_p0)
            assert (wall_1_p0_dist < 1e-3) ^ (wall_2_p0_dist < 1e-3), (
                "line_segments aren't correct for the wall_file_idx!"
                f" Expecting either {wall_1_p0} or {wall_2_p0} to be near {start_wall_pos}"
            )

            wall_file_map[wall_file] = (
                [wall_1, wall_2] if wall_1_p0_dist < 1e-3 else [wall_2, wall_1]
            )

    # NOTE: fix wall bleed issue
    assign_layer_to_rooms(house)

    # NOTE: Scale the hole walls to be proportional to the size of the actual
    # walls. Here, note that the wall size may differ between the line segments
    # and the RoomPlan wall files.
    for wall_file, holes_in_wall in wall_holes.items():
        if len(holes_in_wall) == 0:
            continue

        poly = wall_file_map[wall_file][0]["polygon"]
        expected_wall_length = get_object_pose(wall_file)["bbox"][0]
        actual_wall_length = (
            (poly[0]["x"] - poly[2]["x"]) ** 2 + (poly[0]["z"] - poly[2]["z"]) ** 2
        ) ** 0.5
        length_scale = actual_wall_length / expected_wall_length

        expected_wall_height = get_object_pose(wall_file)["bbox"][1]
        actual_wall_height = abs(poly[0]["y"] - poly[2]["y"])
        assert abs(expected_wall_height - actual_wall_height) < 0.01, (
            f"Wall heights are incorrect! Expected {expected_wall_height}"
            f" but got {actual_wall_height}"
        )

        # scale the hole to the actual wall size
        for i, ((x1, y1), (x2, y2)) in enumerate(holes_in_wall):
            holes_in_wall[i] = (
                (
                    (x1 + expected_wall_length / 2) * length_scale,
                    y1 + expected_wall_height / 2,
                ),
                (
                    (x2 + expected_wall_length / 2) * length_scale,
                    y2 + expected_wall_height / 2,
                ),
            )

    add_windows(
        house=house,
        wall_file_map=wall_file_map,
        wall_holes=wall_holes,
        floor_y=floor_y,
        ceiling_height=ceiling_height,
    )

    add_paintings(controller, house, line_segments)

    # Add obstacles
    event = controller.reset(scene=house)
    thor_objects_metadata = event.metadata["objects"]
    add_obstacle_objects(house=house, thor_objects_metadata=thor_objects_metadata)

    uid = str(uuid.uuid4())
    with open(f"kianas/{uid}.json", "w") as f:
        json.dump(house, f, indent=2)
    # with open("temp.json", "w") as f:
    #     json.dump(house, f, indent=2)

    return None


# controller = Controller(branch="nanna")
# generate_house(controller)
# with open("temp.json", "w") as f:
#     json.dump(house, f, indent=2)
# vis_controller = Controller(
#     branch="naive-scale", platform=CloudRendering, makeAgentsVisible=False
# )
# for _ in range(1):
#     house = generate_house()
#     with open("temp6.json", "w") as f:
#         json.dump(house, f, indent=4, sort_keys=True)

#     # get a timestamp uuid
#     file_id = str(uuid.uuid4())
#     time_stamp = datetime.datetime.now().strftime("%H-%M-%S")
#     uid = time_stamp + "_" + file_id

#     vis_controller.reset(width=2000, height=1500, scene=house)
#     controller.step(action="RandomizeMaterials")
#     vis_controller.step(
#         action="AddThirdPartyCamera",
#         position=dict(x=-2.748, y=1.837, z=1.218),
#         rotation=dict(x=15, y=119.997, z=0),
#         fieldOfView=80,
#     )
#     frame = vis_controller.last_event.third_party_camera_frames[-1].copy()

#     # horizontally flip the frame
#     Image.fromarray(frame).save(f"temp.png")
#     print("saving", uid)


# # TODO: for images, also randomize the object's materials
# # for i in range(10, 100):
# #     print(f"Generating house {i}")
# #     house = generate_house()
# #     with open(f"robothor-rooms-2/{i}.json", "w") as f:
# #         json.dump(house, f, indent=2, sort_keys=True)

# import prior

# d = prior.load_dataset("procthor-10k")["train"]
# d[0]
