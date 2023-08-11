from typing import Dict, List, Tuple

import numpy as np
from phone2sim.utils.polygons import l2_dist


def get_angle_between(p1, p2):
    """Return the angle between two points in degrees."""
    return np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi


# get which line segments are adjacent to each other
def process_line_segments(line_segments):
    graph = []
    for start_line_segment_i, start_line_segment in enumerate(line_segments):
        connected_points_from_line = []
        for start_point in start_line_segment:
            connected_points_g = []
            for end_line_segment_i, end_line_segment in enumerate(line_segments):
                if start_line_segment_i == end_line_segment_i:
                    continue
                for end_point_i, end_point in enumerate(end_line_segment):
                    point_dist = l2_dist(start_point, end_point)
                    # NOTE: 0.16 is the depth of the wall from RoomPlan.
                    # Since some walls are not yet connected, they may be 0.16m apart.
                    if point_dist < 1e-4 or abs(point_dist - 0.16) < 1e-4:
                        connected_points_g.append((end_line_segment_i, end_point_i))
            connected_points_from_line.append(connected_points_g)
        graph.append(connected_points_from_line)

    # set the rotation bias if is is not set
    # TODO: Rotation bias should be based on the mean rotation
    s1_p1, s2_p2 = line_segments[0]
    s1_p1 = np.array(s1_p1)
    s2_p2 = np.array(s2_p2)
    rotation_bias = get_angle_between(s1_p1, s2_p2)

    return graph, rotation_bias


def connect_line_segments(
    line_segments, graph, ideal_wall_rotations: List[float]
) -> None:
    skip_points_g = set()
    for line_i, line in enumerate(line_segments):
        for point_i in range(len(line)):
            if (line_i, point_i) in skip_points_g:
                continue

            # get the points that are all supposed to be the same
            points_graph = graph[line_i][point_i].copy()
            points_graph.append((line_i, point_i))

            # get all the points that connect to any of the same points
            # these are the points on the other sides of the line segments
            # hence the `1 - point_g[1]`.
            connecting_points = [
                (
                    (point_g[0], 1 - point_g[1]),
                    line_segments[point_g[0]][1 - point_g[1]],
                )
                for point_g in points_graph
            ]

            # find the best point based on the ideal wall rotations
            best_point_g = None
            best_score = float("inf")
            for candidate_line_i, candidate_point_i in points_graph:
                point = line_segments[candidate_line_i][candidate_point_i]
                score = sum(
                    abs(
                        (get_angle_between(point, connecting_point) % 90)
                        - ((-ideal_wall_rotations[connecting_point_g[0]]) % 90)
                    )
                    for connecting_point_g, connecting_point in connecting_points
                )
                if score < best_score:
                    best_score = score
                    best_point_g = (candidate_line_i, candidate_point_i)
            assert best_score < 10, (
                f"Closest match: {best_score}deg."
                " Probably a bug in the joining line segment algorithm."
            )

            # set all the points to the best point
            for point_g in points_graph:
                line_segments[point_g[0]][point_g[1]] = line_segments[best_point_g[0]][
                    best_point_g[1]
                ]

            # skip the other point in the line segment since they've been merged
            for candidate_line_i, candidate_point_i in points_graph:
                skip_points_g.add((candidate_line_i, candidate_point_i))


def separate_free_walls(line_segments, graph) -> Dict[int, Tuple[float, float]]:
    """Separate the free walls from the rest of the graph.

    The free walls are the walls that do not form a closed loop.
    """
    skip_over = set()
    free_walls = set()
    all_clean = True
    while all_clean:
        all_clean = False
        for line_i in range(len(graph)):
            for point_i in range(len(graph[line_i])):
                if (line_i, point_i) in skip_over:
                    continue
                if len(graph[line_i][point_i]) == 0:
                    all_clean = True

                    # NOTE: only one connecting point, it does not form a polygon
                    skip_over.add((line_i, point_i))
                    free_walls.add(line_i)
                    for connecting_line_i, connecting_point_i in graph[line_i][
                        1 - point_i
                    ]:
                        graph[connecting_line_i][connecting_point_i].remove(
                            (line_i, 1 - point_i)
                        )

    # remove the free walls from the line segments and graphs
    for line_i in range(len(graph)):
        if line_i in free_walls:
            continue
        for point_i in range(len(graph[line_i])):
            for connection_i, (p1, p2) in enumerate(graph[line_i][point_i]):
                under_p1 = sum([1 for x in free_walls if x < p1])
                under_p2 = sum([1 for x in free_walls if x < p2])
                if under_p1 != 0 or under_p2 != 0:
                    graph[line_i][point_i][connection_i] = (
                        p1 - under_p1,
                        p2 - under_p2,
                    )

    free_wall_segments = {}
    for wall_i in sorted(free_walls, reverse=True):
        free_wall_segments[wall_i] = line_segments[wall_i]
        line_segments.pop(wall_i)
        graph.pop(wall_i)

    return free_wall_segments
