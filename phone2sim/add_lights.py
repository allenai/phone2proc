import json

import numpy as np
from procthor.utils.types import RGB, Light, LightShadow, Vector3
from shapely.geometry import LineString, Polygon
from shapely.ops import split


def add_lights(
    partial_house,
    ceiling_height: float,
    lights_per_room: int,
) -> None:
    """Adds lights to the house.
    Lights include:
    - A point light to the centroid of each room.
    - A directional light.
    Args:
        house: HouseDict, the house to add lights to.
    """
    floor_polygons = {}
    for room in partial_house["rooms"]:
        poly = Polygon([(point["x"], point["z"]) for point in room["floorPolygon"]])
        floor_polygons[room["id"]] = poly

    # add directional light
    lights = []

    # add point lights
    for i, room in enumerate(partial_house["rooms"]):
        room_id = room["id"]
        poly = floor_polygons[room_id]

        min_x = poly.bounds[0]
        max_x = poly.bounds[2]
        min_z = poly.bounds[1]
        max_z = poly.bounds[3]
        x_size = max_x - min_x
        z_size = max_z - min_z
        x_size, z_size

        polys = [poly]
        if lights_per_room > 1:
            if x_size > z_size:
                x_chops = np.linspace(0, x_size, lights_per_room - 1 + 2)[1:-1]
                lines = [
                    LineString(
                        [(min_x + x_chop, min_z - 1), (min_x + x_chop, max_z + 1)]
                    )
                    for x_chop in x_chops
                ]
            else:
                z_chops = np.linspace(0, z_size, lights_per_room - 1 + 2)[1:-1]
                lines = [
                    LineString(
                        [(min_x - 1, min_z + z_chop), (max_x + 1, min_z + z_chop)]
                    )
                    for z_chop in z_chops
                ]

            for line in lines:
                new_polys = []
                for poly in polys:
                    # get the 2 separate polygons after the split
                    new_polys.extend(split(poly, line).geoms)
                polys = new_polys

        for j, poly in enumerate(polys):
            x = poly.centroid.x
            z = poly.centroid.y

            # NOTE: The point lights may be overwritten by the skybox.
            lights.append(
                Light(
                    id=f"light_{i}_{j}",
                    type="point",
                    position=Vector3(x=x, y=ceiling_height - 0.2, z=z),
                    intensity=0.75,
                    range=15,
                    rgb=RGB(r=1.0, g=0.855, b=0.722),
                    shadow=LightShadow(
                        type="Soft",
                        strength=1,
                        normalBias=0,
                        bias=0.05,
                        nearPlane=0.2,
                        resolution="FromQualitySettings",
                    ),
                    roomId=room_id,
                )
            )

    partial_house["proceduralParameters"]["lights"] = lights
