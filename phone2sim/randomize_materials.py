import random

from procthor.databases import material_database, solid_wall_colors

FLOOR_MATERIALS = material_database["Wood"]
WALL_MATERIALS = material_database["Wall"]

P_SAMPLE_SOLID_WALL_COLOR = 0.5
"""Probability of sampling a solid wall color instead of a material."""

P_ALL_WALLS_SAME = 0.35
"""Probability that all wall materials are the same."""

P_ALL_FLOOR_SAME = 0.15
"""Probability that all floor materials are the same."""


def sample_wall_material():
    if random.random() < P_SAMPLE_SOLID_WALL_COLOR:
        return {
            "color": random.choice(solid_wall_colors),
            "name": "PureWhite",
        }
    return {
        "name": random.choice(WALL_MATERIALS),
    }


def randomize_wall_materials(house) -> None:
    """Randomize the materials on each wall."""
    if random.random() < P_ALL_WALLS_SAME:
        wall_material = sample_wall_material()
        for wall in house["walls"]:
            wall["material"] = wall_material

        # NOTE: set the ceiling
        house["proceduralParameters"]["ceilingMaterial"] = wall_material
        return

    # NOTE: independently randomize each room's materials.
    room_ids = set()
    for wall in house["walls"]:
        room_ids.add(wall["roomId"])
    room_ids.add("ceiling")

    wall_materials_per_room = dict()
    for room_id in room_ids:
        wall_materials_per_room[room_id] = sample_wall_material()

    for wall in house["walls"]:
        wall["material"] = wall_materials_per_room[wall["roomId"]]

    # NOTE: randomize ceiling material
    house["proceduralParameters"]["ceilingMaterial"] = wall_materials_per_room[
        "ceiling"
    ]


def randomize_floor_materials(house) -> None:
    """Randomize the materials on each floor."""
    if random.random() < P_ALL_FLOOR_SAME:
        floor_material = random.choice(FLOOR_MATERIALS)
        for room in house["rooms"]:
            room["floorMaterial"] = {"name": floor_material}
        return

    for room in house["rooms"]:
        room["floorMaterial"] = {"name": random.choice(FLOOR_MATERIALS)}


def randomize_wall_and_floor_materials(house) -> None:
    """Randomize the materials on each wall and floor."""
    randomize_wall_materials(house)
    randomize_floor_materials(house)
