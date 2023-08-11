"""Get the polygons that make up the line segments."""
from typing import List, Optional, Tuple, Union

import numpy as np
from shapely.geometry import MultiPolygon, Polygon


def get_angle(x, y):
    angle = np.arctan2(y, x) * 180 / np.pi
    if angle < 0:
        angle += 360
    return angle


def get_clockwise_angle(x1: float, y1: float, x2: float, y2: float) -> float:
    angle1 = get_angle(x=x1, y=y1)
    angle2 = get_angle(x=x2, y=y2)
    return (angle1 - angle2) % 360


def l2_dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def serialize_point(p) -> Tuple[float, float]:
    return (round(p[0], 3), round(p[1], 3))


def get_polygon(polygon_points) -> Polygon:
    serial_max_poly_points = [serialize_point(p) for p in polygon_points]
    return Polygon(serial_max_poly_points)


def get_nearest_coord_dist(point, polygon: Union[Polygon, MultiPolygon]) -> float:
    x = []
    y = []
    if isinstance(polygon, Polygon):
        x, y = polygon.exterior.xy
    elif isinstance(polygon, MultiPolygon):
        for p in polygon.geoms:
            x_, y_ = p.exterior.xy
            x.extend(x_)
            y.extend(y_)

    min_dist = np.inf
    for x_, y_ in zip(x, y):
        dist = l2_dist(point, (x_, y_))
        min_dist = min(min_dist, dist)
    return min_dist


def get_inner_polygon(
    start_i: int,
    start_p_i: int,
    graph,
    line_segments,
    get_max_poly: bool = False,
    existing_poly: Optional[Polygon] = None,
) -> List[Tuple[float, float]]:
    """
    polygon_graph represents the wall_file_index of each of the walls in the
    polygon. This index is the same as the line_segments and graph index.
    """
    polygon = [(line_segments[start_i][start_p_i])]
    polygon_graph = [(start_i, start_p_i)]

    next_points_from_end = graph[start_i][start_p_i]

    min_connecting_point_i = None
    min_angle = float("inf")
    for i, p_i in next_points_from_end:
        connecting_point = line_segments[i][1 - p_i]

        # NOTE: check if the distance between the connecting point and the
        # existing polygon is small enough.
        if existing_poly is not None:
            nearest_coord_dist = get_nearest_coord_dist(connecting_point, existing_poly)
            if nearest_coord_dist > 1e-2:
                continue

        angle = get_clockwise_angle(
            x1=-1,
            y1=0,
            x2=connecting_point[0] - polygon[-1][0],
            y2=connecting_point[1] - polygon[-1][1],
        )
        if angle < min_angle:
            min_angle = angle
            min_connecting_point_i = (i, 1 - p_i)

    next_point_from_line = line_segments[start_i][1 - start_p_i]
    consider_point_from_line = True
    if existing_poly is not None:
        nearest_coord_dist = get_nearest_coord_dist(next_point_from_line, existing_poly)
        if nearest_coord_dist > 1e-2:
            consider_point_from_line = False
    if consider_point_from_line:
        angle_from_line = get_clockwise_angle(
            x1=-1,
            y1=0,
            x2=next_point_from_line[0] - polygon[-1][0],
            y2=next_point_from_line[1] - polygon[-1][1],
        )
        if angle_from_line < min_angle:
            min_connecting_point_i = start_i, 1 - start_p_i
            min_angle = angle_from_line

    polygon.append(line_segments[min_connecting_point_i[0]][min_connecting_point_i[1]])
    polygon_graph.append(min_connecting_point_i[0])

    point_i, point_p_i = min_connecting_point_i
    while l2_dist(polygon[-1], polygon[0]) > 1e-5:
        if len(polygon) > 100:
            raise Exception("Probably in an infinite loop!")
        next_points = graph[point_i][point_p_i]
        if len(next_points) == 1:
            next_point = next_points[0]
        else:
            # NOTE: get the maximum clockwise angle
            max_angle = float("inf" if get_max_poly else "-inf")
            max_connecting_point_i = None
            for i, p_i in next_points:
                connecting_point = line_segments[i][1 - p_i]
                angle = get_clockwise_angle(
                    x1=polygon[-2][0] - polygon[-1][0],
                    y1=polygon[-2][1] - polygon[-1][1],
                    x2=connecting_point[0] - polygon[-1][0],
                    y2=connecting_point[1] - polygon[-1][1],
                )

                if angle == max_angle:
                    raise Exception("Found two points with the same angle!")

                if (get_max_poly and angle < max_angle) or (
                    not get_max_poly and angle > max_angle
                ):
                    max_angle = angle
                    max_connecting_point_i = (i, p_i)
            next_point = max_connecting_point_i

        point_i, point_p_i = next_point
        point_p_i = 1 - point_p_i
        polygon.append(line_segments[point_i][point_p_i])
        polygon_graph.append(next_point[0])

    return polygon[:-1], [polygon_graph[-1]] + polygon_graph[1:-1]


def get_inner_polygons(line_segments, graph):
    if len(line_segments) == 0:
        return [], []
    import pickle

    with open("graph.pkl", "wb") as f:
        pickle.dump(graph, f)
    with open("line_segments.pkl", "wb") as f:
        pickle.dump(line_segments, f)

    min_point = (float("inf"), float("inf"))
    min_point_i = None
    for i, (p1, p2) in enumerate(line_segments):
        for p_i, p in enumerate([p1, p2]):
            if p[0] < min_point[0] or (p[0] == min_point[0] and p[1] < min_point[1]):
                min_point = p
                min_point_i = (i, p_i)

    polygons = []
    polygon_graphs = []

    max_polygon_points, _ = get_inner_polygon(
        *min_point_i, graph, line_segments, get_max_poly=True
    )
    max_poly = get_polygon(max_polygon_points)

    while max_poly.area > 1e-3:
        polygon_points, poly_graph = get_inner_polygon(
            *min_point_i,
            graph,
            line_segments,
            get_max_poly=False,
            existing_poly=max_poly
        )
        poly = get_polygon(polygon_points)
        polygons.append(polygon_points)
        polygon_graphs.append(poly_graph)

        max_poly = max_poly.difference(poly)

        # get the polygons from the multipolygon
        if isinstance(max_poly, MultiPolygon):
            x = []
            y = []
            for p in max_poly.geoms:
                x.extend(p.exterior.coords.xy[0])
                y.extend(p.exterior.coords.xy[1])
        else:
            x, y = max_poly.exterior.coords.xy
        min_point = (float("inf"), float("inf"))
        for coord in list(zip(x, y)):
            if coord[0] < min_point[0] or (
                coord[0] == min_point[0] and coord[1] < min_point[1]
            ):
                min_point = coord

        for i, (p1, p2) in enumerate(line_segments):
            for p_i, p in enumerate([p1, p2]):
                if l2_dist(p, min_point) < 5e-3:
                    min_point_i = (i, p_i)
                    break
            else:
                continue
            break
    return polygons, polygon_graphs
