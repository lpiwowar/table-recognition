import multiprocessing
from math import floor

import cv2
import numpy as np
from skimage.draw import line_nd

from table_recognition.graph.edge_discovery.edge import Edge
from table_recognition.graph.node import Node


class NodeVisibility(object):

    SAMPLING_RATE = 10  # Create ray every SAMPLING_RATE degrees
    WINDOW_SIZE = 30    # How big is the window we sample from

    def __init__(self, graph):
        self.graph = graph

        img = cv2.imread(self.graph.img_path)
        self.img_h, self.img_w, self.img_c = img.shape

        self.nodes_db = {}

    def populate_nodes_db(self):
        for node in self.graph.nodes:
            self.nodes_db[node.id] = node

    def discover_edges_subprocess(self, edges, start, end):
        self.populate_nodes_db()
        boxes_image = self.render_boxes_image()
        nodes = list(self.graph.nodes)

        for node in nodes[start:end]:
            # node_distance = { <node_id>: <distance> }
            node_degrees = {}
            for degree in range(0, 181, NodeVisibility.SAMPLING_RATE):
                # Get values from the image that are on the line
                line_coordinates = self.get_line_coordinates(node.bbox["center"], degree)
                line_values = boxes_image[line_coordinates[0], line_coordinates[1]].astype(int).squeeze()

                # Find indexes that split the line into two sections
                node_values_idxs = np.where(line_values == (node.id + 1))[0]
                if node_values_idxs.size == 0:
                    # This may happen in rare cases (e.g.: when two boxes overlap each other a lot!)
                    continue
                node_values_idxs_min = np.min(node_values_idxs)
                node_values_idxs_max = np.max(node_values_idxs)

                # Find the first intersection on the right/left
                line_values_right = line_values[(node_values_idxs_max + 1):]
                line_values_left = np.flip(line_values[:node_values_idxs_min])

                right_nonzero_idx = np.nonzero(line_values_right)[0]
                left_nonzero_idx = np.nonzero(line_values_left)[0]

                right_nonzero_idx = right_nonzero_idx[0] if len(right_nonzero_idx) >= 1 else None
                left_nonzero_idx = left_nonzero_idx[0] if len(left_nonzero_idx) >= 1 else None

                if right_nonzero_idx is not None:
                    right_node_id = line_values_right[right_nonzero_idx] - 1
                    bin_id = degree // NodeVisibility.WINDOW_SIZE
                    node_degrees[bin_id] = node_degrees.get(bin_id, [])
                    node_degrees[bin_id] += [(right_node_id, right_nonzero_idx)]

                if left_nonzero_idx is not None:
                    left_node_id = line_values_left[left_nonzero_idx] - 1
                    new_degrees = 180 + degree
                    bin_id = new_degrees // 30
                    node_degrees[bin_id] = node_degrees.get(bin_id, [])
                    node_degrees[bin_id] += [(left_node_id, left_nonzero_idx)]

            for key in node_degrees:
                node_degrees[key].sort(key=lambda item: item[1], reverse=True)

            for key in node_degrees:
                if node_degrees[key]:
                    # print(node_degrees[key])
                    new_id = node_degrees[key].pop()[0]
                    edges += [Edge(node, self.nodes_db[new_id])]
                    edges += [Edge(self.nodes_db[new_id], node)]
            # node_distance_sorted = sorted(node_degrees.items(), key=lambda item: item[1])
            # for node_id, _ in node_distance_sorted[:NodeVisibility.K_NEAREST_VALUES]:
            #     edges += [Edge(node, self.nodes_db[node_id])]
            #     edges += [Edge(self.nodes_db[node_id], node)]

        edges = {edge for edge in edges if not edge.is_reflexive()}

    def discover_edges(self):
        # manager = multiprocessing.Manager()
        # global_edges = manager.list()  # Shared variable - all discovered edges
        # global_edges = []
        # self.discover_edges_subprocess(global_edges, 0, len(self.graph.nodes))

        # Define jobs for each subprocess
        proc1_start = 0
        proc1_end = len(self.graph.nodes) // 3
        proc2_start = (len(self.graph.nodes) // 3) + 1
        proc2_end = 2 * (len(self.graph.nodes) // 3)
        proc3_start = (2 * (len(self.graph.nodes) // 3)) + 1
        proc3_end = len(self.graph.nodes)

        # Define subprocesses
        proc1 = multiprocessing.Process(target=self.discover_edges_subprocess, args=(global_edges, proc1_start,
                                                                                     proc1_end))
        # proc1 = multiprocessing.Process(target=self.discover_edges_subprocess, args=(global_edges, 0,
        #                                                                               len(self.graph.nodes)))
        proc2 = multiprocessing.Process(target=self.discover_edges_subprocess, args=(global_edges, proc2_start,
                                                                                     proc2_end))
        proc3 = multiprocessing.Process(target=self.discover_edges_subprocess, args=(global_edges, proc3_start,
                                                                                     proc3_end))

        # Start subprocesses
        proc1.start()
        proc2.start()
        proc3.start()

        # Wait for subprocesses to finish
        proc1.join()
        proc2.join()
        proc3.join()

        self.graph.edges = set(global_edges)

    def get_line_coordinates(self, point, angle_deg):
        assert 0 <= point[0] <= self.img_w, "ERROR: Coordinates out of image"
        assert 0 <= point[1] <= self.img_h, "ERROR: Coordinates out of image"

        x, y = point

        # Return line immediately for extreme values 0, 90, and 180 degrees
        if angle_deg in [0, 90, 180]:
            if angle_deg in [0, 180]:
                # Vertical line
                line_coords = line_nd((0, x), (self.img_h - 1, x))
            else:
                # Horizontal line
                line_coords = line_nd((y, 0), (y, self.img_w - 1))
            line_array = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
            line_array[line_coords] = 255

            line_array = np.nonzero(line_array)
            line_array = np.dstack((line_array[0], line_array[1]))[0]
            line_array_idx = np.lexsort((line_array[:, 0], line_array[:, 1]))
            line_array = line_array[line_array_idx]
            (line_array_y, line_array_x) = np.split(line_array, 2, axis=1)

            return np.array([line_array_y, line_array_x])

        # angle_rad = (np.pi / 180) * angle_deg
        angle_rad = np.radians(angle_deg)

        # Calculate parameters of the line (weight, bias)
        line_slope = np.tan(-((np.pi / 2) - angle_rad))
        line_bias = y - line_slope * x

        # Calculate intersections with edges of the image
        x_top_value = (self.img_h - line_bias) / line_slope
        x_bot_value = (0 - line_bias) / line_slope
        y_right_value = self.img_w * line_slope + line_bias
        y_left_value = 0 * line_slope + line_bias

        x_top = floor(x_top_value) if 0 <= x_top_value <= self.img_w else None
        x_bot = floor(x_bot_value) if 0 <= x_bot_value <= self.img_w else None
        y_right = floor(y_right_value) if 0 <= y_right_value <= self.img_h else None
        y_left = floor(y_left_value) if 0 <= y_left_value <= self.img_h else None

        # Find the two points
        line_points = []
        line_points += [(self.img_h - 1, x_top)] if x_top is not None else []
        line_points += [(0, x_bot)] if x_bot is not None else []
        line_points += [(y_right, self.img_w - 1)] if y_right is not None else []
        line_points += [(y_left, 0)] if y_left is not None else []

        line_coords = line_nd(line_points[0], line_points[1])
        line_array = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
        line_array[line_coords] = 255

        line_array = np.nonzero(line_array)
        line_array = np.dstack((line_array[0], line_array[1]))[0]
        line_array_idx = np.lexsort((line_array[:, 0], line_array[:, 1]))
        line_array = line_array[line_array_idx]
        (line_array_y, line_array_x) = np.split(line_array, 2, axis=1)

        return np.array([line_array_y, line_array_x])

    def render_boxes_image(self):
        """Generate 2D-array in which each pixels says id of node it represents"""
        render_image = np.zeros((self.img_h, self.img_w))
        for node in self.graph.nodes:
            (min_x, min_y, max_x, max_y) = node.bbox["rtree"]
            render_image[min_y:max_y, min_x:max_x] = node.id + 1   # Beware: This is here because of np.nonzero()
            cv2.putText(render_image, f"{node.id + 1}", node.bbox["center"], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1,
                        cv2.LINE_AA)

        return render_image
