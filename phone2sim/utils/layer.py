# removes the light bleed issue using the four color theorem

import itertools
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List

import networkx as nx
import pysat.solvers

if TYPE_CHECKING:
    from procthor.generation import PartialHouse

from shapely.geometry import Point, Polygon


def get_room_to_layer_map(house):
    open_wall_connections = defaultdict(set)
    for wall in house["walls"]:
        if "empty" in wall and wall["empty"]:
            room_id = wall["id"][len("wall|") :]
            room_id = int(room_id[: room_id.find("|")])

            wall_pos_id = wall["id"][len("wall|") :]
            wall_pos_id = wall_pos_id[wall_pos_id.find("|") + 1 :]
            open_wall_connections[wall_pos_id].add(room_id)

    connection_pairs = [{r1, r2} for r1, r2 in open_wall_connections.values()]
    connections = []
    while connection_pairs:
        connection = connection_pairs[0]
        offset = 0
        for i, oc in enumerate(connection_pairs[1:].copy()):
            r1, r2 = list(oc)
            if r1 in connection or r2 in connection:
                connection.update(oc)
                del connection_pairs[i - offset]
                offset += 1
        del connection_pairs[0]
        connections.append(connection)

    # Maps the key in the graph to the set of room ids that are assigned to it
    open_room_connection_keys = {
        -(i + 1): connection for i, connection in enumerate(connections)
    }

    # Maps the room id to the key its open neighbor group is assigned in the graph
    room_id_to_open_key = {
        room_id: key
        for key, room_ids in open_room_connection_keys.items()
        for room_id in room_ids
    }

    pos_id_to_room_ids = defaultdict(set)
    for wall in house["walls"]:
        room_id = wall["id"].split("|")[1]
        if room_id in {"exterior", "ceiling"}:
            continue
        room_id = int(room_id)
        pos_id = "|".join(wall["id"].split("|")[2:6])
        pos_id_to_room_ids[pos_id].add(room_id)
    adjacent_rooms = set(
        [tuple(sorted(list(x))) for x in pos_id_to_room_ids.values() if len(x) == 2]
    )
    room_neighbors = defaultdict(set)
    for room_0, room_1 in adjacent_rooms:
        if room_0 in room_id_to_open_key:
            room_0 = room_id_to_open_key[room_0]
        if room_1 in room_id_to_open_key:
            room_1 = room_id_to_open_key[room_1]

        if room_0 != room_1:
            room_neighbors[room_0].add(room_1)
            room_neighbors[room_1].add(room_0)

    graph = nx.Graph()
    for node, neighbors in room_neighbors.items():
        graph.add_node(node)
        for neighbor in neighbors:
            if neighbor not in graph:
                graph.add_node(neighbor)
            graph.add_edge(node, neighbor)

    room_to_layer_map = four_color_graph(graph)
    if not room_to_layer_map:
        return {}

    for room_id, open_key in room_id_to_open_key.items():
        room_to_layer_map[room_id] = room_to_layer_map[open_key]

    for open_key in set(room_id_to_open_key.values()):
        del room_to_layer_map[open_key]

    return room_to_layer_map


def assign_room_to_layer(house, room_to_layer_map):
    poly_map = {}
    room_id_map = {
        f"room|{room_i}": layer for room_i, layer in room_to_layer_map.items()
    }

    # NOTE: Map the rooms
    for room in house["rooms"]:
        layer = room_id_map[room["id"]]
        room["layer"] = f"Procedural{layer}"
        poly_map[room["id"]] = Polygon([(p["x"], p["z"]) for p in room["floorPolygon"]])

    # NOTE: Map the lights
    lights = house["proceduralParameters"]["lights"]
    for light in lights:
        layer = room_id_map[light["roomId"]]
        light["layer"] = f"Procedural{layer}"
        light["cullingMaskOff"] = [
            f"Procedural{i}" for i in list(set(range(4)) - {layer})
        ]

    # NOTE: Map the walls
    back_wall_room_map = {
        "|".join(wall["id"].split("|")[2:]): wall["roomId"]
        for wall in house["walls"]
        if wall["roomId"] is not None
    }
    for wall in house["walls"]:
        if wall["roomId"] is None:
            # look for the wall on the other side and use it
            wall_pos_id = "|".join(wall["id"].split("|")[2:])
            wall["layer"] = f"Procedural{room_id_map[back_wall_room_map[wall_pos_id]]}"
        else:
            wall["layer"] = f"Procedural{room_id_map[wall['roomId']]}"

    # NOTE: Map the objects
    def set_object_layers(objects: List[Dict[str, Any]]) -> None:
        for obj in objects:
            point = Point(obj["position"]["x"], obj["position"]["z"])
            for room_id, poly in poly_map.items():
                if poly.contains(point):
                    layer = room_id_map[room_id]
                    obj["layer"] = f"Procedural{layer}"
                    break
            else:
                print(f"Object {obj} not in any room")
            # TODO: layer assignment is wrong
            if "children" in obj and obj["children"]:
                set_object_layers(obj["children"])

    set_object_layers(house["objects"])


def four_color_graph(graph: nx.Graph):
    node_to_vars = {}
    vars_to_node = {}
    k = 1
    sat = pysat.solvers.Glucose3()
    for node in graph.nodes:
        node_to_vars[node] = list(range(k, k + 4))
        k += 4
        for i in node_to_vars[node]:
            vars_to_node[i] = node

        sat.add_clause(node_to_vars[node])

        for i, j in itertools.combinations(node_to_vars[node], 2):
            for l in node_to_vars[node]:
                if i != l:
                    sat.add_clause([-i, -l])

    for u, v in graph.edges:
        for i, j in zip(node_to_vars[u], node_to_vars[v]):
            sat.add_clause([-i, -j])

    sat.solve()

    node_to_color = {}
    for result in sat.get_model():
        if result > 0:
            node_to_color[vars_to_node[result]] = result % 4

    return node_to_color


def assign_layer_to_rooms(house) -> None:
    """Assigns layers to each room.

    Used to avoid light bleeding through the walls.
    """
    room_to_layer_map = get_room_to_layer_map(house)
    if not room_to_layer_map:
        assert len(house["doors"]) <= 1
    else:
        assert len(room_to_layer_map) == len(house["rooms"])
        assign_room_to_layer(house, room_to_layer_map)
