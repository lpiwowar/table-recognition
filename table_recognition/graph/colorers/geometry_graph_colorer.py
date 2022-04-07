import cv2
import math
import numpy as np
import sys

import alphashape
from rtree import index
from shapely.geometry import Polygon
from shapely.geometry import LineString
from shapely.ops import split

from table_recognition.graph.colorers.utils import get_multiple_values_from_dict


class GeometryGraphColorer(object):
    def __init__(self, graph):
        self.graph = graph
        self.img_height, self.img_width, _ = cv2.imread(self.graph.img_path).shape

        self.rtree_index = index.Index()
        self.rtree_index_2_node = {}

    def color_graph(self):
        self.build_rtree()
        self.color_nodes()
        self.color_edges()

    def color_nodes(self):
        for node in self.graph.nodes:
            # Bounding box center
            x, y = node.bbox["center"]
            position = [x / self.img_width, y / self.img_height]

            # Bounding box width and height
            [(min_x, min_y), (max_x, max_y)] = node.bbox["corners"]
            bbox_width = abs(max_x - min_x) / self.img_width
            bbox_height = abs(max_y - min_y) / self.img_height
            bbox_dimensions = [bbox_width, bbox_height]

            node.input_feature_vector = position + bbox_dimensions

    def color_edges(self):
        # TODO - Cleanup edge coloring for GeometryGraphColorer
        for edge in self.graph.edges:
            # Center of node1
            node1_x, node1_y = edge.node1.bbox["center"]
            node1_x, node1_y = node1_x / self.img_width, node1_y / self.img_height

            # Center of node2
            node2_x, node2_y = edge.node2.bbox["center"]
            node2_x, node2_y = node2_x / self.img_width, node2_y / self.img_height

            # -- Feature1: Distance of the two centers -----------------------------------------------------------------
            distance = np.linalg.norm(np.array([node1_x, node1_y]) - np.array([node2_x, node2_y]))
            # ----------------------------------------------------------------------------------------------------------

            # -- Feature2: Average of the two centers ------------------------------------------------------------------
            avg_position_x, avg_position_y = (node1_x + node2_x) / 2, (node1_y + node2_y) / 2
            # ----------------------------------------------------------------------------------------------------------

            # -- Feature4: Orientation of the edge ---------------------------------------------------------------------
            right_node = edge.node1 if edge.node1.bbox["center"][0] > edge.node2.bbox["center"][0] else edge.node2
            left_node = edge.node2 if edge.node1.bbox["center"][0] > edge.node2.bbox["center"][0] else edge.node1
            right_node_x, right_node_y = right_node.bbox["center"]
            left_node_x, left_node_y = left_node.bbox["center"]

            x_distance = right_node_x - left_node_x
            y_distance = abs(right_node_y - left_node_y)
            orientation = y_distance / (x_distance + sys.float_info.epsilon)
            orientation = math.degrees(math.atan(orientation))

            if (right_node_y - left_node_y) > 0:
                orientation = 90 + (90 - orientation)

            if orientation < 5 or orientation > 175:
                orientation = 180
            # ----------------------------------------------------------------------------------------------------------

            # -- Feature5: Vertical and horizontal overlap ------------------------------------------------------------
            [(node1_min_x, node1_min_y), (node1_max_x, node1_max_y)] = edge.node1.bbox["corners"]
            [(node2_min_x, node2_min_y), (node2_max_x, node2_max_y)] = edge.node2.bbox["corners"]

            # a) vertical overlap
            x_max = min(node1_max_x, node2_max_x)
            x_min = max(node1_min_x, node2_min_x)
            x_overlap = (x_max - x_min) if (x_max - x_min) > 0 else 0

            # b) horizontal overlap
            y_max = min(node1_max_y, node2_max_y)
            y_min = max(node1_min_y, node2_min_y)
            y_overlap = (y_max - y_min) if (y_max - y_min) > 0 else 0
            # ---------------------------------------------------------------------------------------------------------

            # -- Feature5: Calculate overlap in percentage ------------------------------------------------------------
            if x_overlap > 0:
                ys = [node1_min_y, node2_min_y, node1_max_y, node2_max_y]
                ys.sort()
                bbox_in_between = (x_min, ys[1], x_max, ys[2])
                nodes_intersections_idx = list(self.rtree_index.intersection(bbox_in_between))

                if edge.node1.id in nodes_intersections_idx:
                    nodes_intersections_idx.remove(edge.node1.id)
                if edge.node2.id in nodes_intersections_idx:
                    nodes_intersections_idx.remove(edge.node2.id)
                nodes = get_multiple_values_from_dict(self.rtree_index_2_node, nodes_intersections_idx)

                bbox_in_between_set = set(range(x_min, x_max+1))
                for node in nodes:
                    iter_node_x_max = node.bbox["rtree"][2]
                    iter_node_x_min = node.bbox["rtree"][0]
                    iter_node_set = set(range(iter_node_x_min, iter_node_x_max+1))
                    bbox_in_between_set = bbox_in_between_set - iter_node_set

                bbox_in_between_list = list(bbox_in_between_set)
                bbox_in_between_list.sort()

                if len(bbox_in_between_list) <= 0:
                    x_overlap = 0
                else:
                    x_overlap = bbox_in_between_list[-1] - bbox_in_between_list[0]
                x_min_side = min(abs(node1_max_x - node1_min_x), abs(node2_max_x - node2_min_x))
                x_overlap = x_overlap / x_min_side

                if (x_max - x_min) < 0:
                    x_overlap = 0

            if y_overlap > 0:
                xs = [node1_min_x, node2_min_x, node1_max_x, node2_max_x]
                xs.sort()
                bbox_in_between = (xs[1], y_min, xs[2], y_max)

                nodes_intersections_idx = list(self.rtree_index.intersection(bbox_in_between))
                if edge.node1.id in nodes_intersections_idx:
                    nodes_intersections_idx.remove(edge.node1.id)
                if edge.node2.id in nodes_intersections_idx:
                    nodes_intersections_idx.remove(edge.node2.id)
                nodes = get_multiple_values_from_dict(self.rtree_index_2_node, nodes_intersections_idx)

                bbox_in_between_set = set(range(y_min, y_max + 1))

                for node in nodes:
                    iter_node_y_max = node.bbox["rtree"][3]
                    iter_node_y_min = node.bbox["rtree"][1]
                    iter_node_set = set(range(iter_node_y_min, iter_node_y_max + 1))
                    bbox_in_between_set = bbox_in_between_set - iter_node_set

                bbox_in_between_list = list(bbox_in_between_set)
                bbox_in_between_list.sort()

                if len(bbox_in_between_list) <= 0:
                    y_overlap = 0
                else:
                    y_overlap = bbox_in_between_list[-1] - bbox_in_between_list[0]

                y_min_side = min(abs(node1_max_y - node1_min_y), abs(node2_max_y - node2_min_y))
                y_overlap = y_overlap / y_min_side
            # ----------------------------------------------------------------------------------------------------------

            # polygons_see_each_other = int(self.polygons_see_each_other(edge.node1, edge.node2))

            edge.input_feature_vector = [float(distance)] + \
                                        [avg_position_x, avg_position_y] + \
                                        [orientation] + \
                                        [x_overlap, y_overlap]
                                        # [polygons_see_each_other]

    def build_rtree(self):
        for node in self.graph.nodes:
            self.rtree_index.insert(node.id, node.bbox["rtree"])
            self.rtree_index_2_node[node.id] = node

    def polygons_see_each_other(self, node1, node2):
        """
        Function that checks whether two polygons see each other (meaning there is not
        a point in polygon1 nor polygon2 from which it is possible to "see" any point
        from the other polygon)

        :param node1: Points that define the first polygon
        :param node2: Points that define the second polygon
        :return: True if polygons see each other
                 False if polygons do not see each other
        """
        polygon1 = node1.bbox["polygon"]
        polygon2 = node2.bbox["polygon"]

        # Create polygon that represents the convex hull of polygon1 and polygon2
        hull_points = polygon1 + polygon2

        # Source:
        # https://stackoverflow.com/questions/10846431/ordering-shuffled-points-that-can-be-joined-to-form-a-polygon-in-python
        alpha_shape = alphashape.alphashape(hull_points, 0.5)
        hull_polygon = Polygon(alpha_shape)

        max_x = max(hull_points, key=lambda item: item[0])[0]
        max_y = max(hull_points, key=lambda item: item[1])[1]
        min_x = min(hull_points, key=lambda item: item[0])[0]
        min_y = min(hull_points, key=lambda item: item[1])[1]
        hull_points_bbox = (min_x, min_y, max_x, max_y)

        nodes_test = list(self.rtree_index.intersection((0, 0, 10000, 10000)))

        # Find nodes that intersect created hull
        nodes_intersections_idx = list(self.rtree_index.intersection(hull_points_bbox))
        if node1.id in nodes_intersections_idx:
            nodes_intersections_idx.remove(node1.id)
        if node2.id in nodes_intersections_idx:
            nodes_intersections_idx.remove(node2.id)

        nodes = get_multiple_values_from_dict(self.rtree_index_2_node, nodes_intersections_idx)

        for node in nodes:
            # Split the hull polygon using the node polygon
            node_linestring = LineString(node.bbox["polygon"] + [node.bbox["polygon"][0]])
            hull_polygon_splitted = split(hull_polygon, node_linestring).geoms

            # Check how many new polygons were created by splitting the hull_polygon
            if len(hull_polygon_splitted) >= 2:
                # If there are more than 2 new polygons => the nodes do not see each other
                return False
            elif len(hull_polygon_splitted) <= 1:
                continue
            else:
                # Else find the cut hull polygon in the output of split() and continue the process
                node_polygon = Polygon(node.bbox["polygon"])
                fst_intersection = node_polygon.intersection(hull_polygon_splitted[0]).area / node_polygon.area
                hull_polygon = hull_polygon_splitted[0] if fst_intersection > 0.9 else hull_polygon_splitted[1]

        return True
