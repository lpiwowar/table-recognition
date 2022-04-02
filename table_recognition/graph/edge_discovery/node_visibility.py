from math import floor

import cv2
import numpy as np

from skimage.draw import line_nd

from table_recognition.graph.edge_discovery.edge import Edge
from table_recognition.graph.node import Node


class NodeVisibility(object):

    K_NEAREST_VALUES = 4
    SAMPLING_RATE = 10

    def __init__(self, graph):
        self.graph = graph

        img = cv2.imread(self.graph.img_path)
        self.img_h, self.img_w, self.img_c = img.shape

        self.nodes_db = {}

    def populate_nodes_db(self):
        for node in self.graph.nodes:
            self.nodes_db[node.id] = node

    def discover_edges(self):
        self.populate_nodes_db()
        boxes_image = self.render_boxes_image()

        print(f"discovering edges for: {self.graph.img_path}")

        for node in self.graph.nodes:
            # node_distance = { <node_id>: <distance> }
            node_distance = {}
            for degree in range(0, 181, NodeVisibility.SAMPLING_RATE):
                # Get values from the image that are on the line
                line_coordinates = self.get_line_coordinates(node.bbox["center"], degree)
                line_values = boxes_image[line_coordinates[0], line_coordinates[1]].astype(int).squeeze()

                # Find indexes that split the line into two sections
                node_values_idxs = np.where(line_values == node.id)
                node_values_idxs_min = np.min(node_values_idxs)
                node_values_idxs_max = np.max(node_values_idxs)

                # Find the first intersection on the right/left
                line_values_right = line_values[(node_values_idxs_max+1):]
                line_values_left = np.flip(line_values[:node_values_idxs_min])

                right_nonzero_idx = np.nonzero(line_values_right)[0]
                left_nonzero_idx = np.nonzero(line_values_left)[0]

                right_nonzero_idx = right_nonzero_idx[0] if len(right_nonzero_idx) >= 1 else None
                left_nonzero_idx = left_nonzero_idx[0] if len(left_nonzero_idx) >= 1 else None

                if right_nonzero_idx:
                    right_node_id = line_values_right[right_nonzero_idx]
                    node_distance[right_node_id] = right_nonzero_idx

                if left_nonzero_idx:
                    left_node_id = line_values_left[left_nonzero_idx]
                    node_distance[left_node_id] = left_nonzero_idx

            node_distance_sorted = sorted(node_distance.items(), key=lambda item: item[1])
            for node_id, _ in node_distance_sorted[:NodeVisibility.K_NEAREST_VALUES]:
                self.graph.edges = self.graph.edges.union({Edge(node, self.nodes_db[node_id])})
                self.graph.edges = self.graph.edges.union({Edge(self.nodes_db[node_id], node)})

        self.graph.edges = {edge for edge in self.graph.edges if not edge.is_reflexive()}

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
        line_slope = np.tan(-((np.pi/2) - angle_rad))
        line_bias = y - line_slope * x

        # Calculate intersections with edges of the image
        x_top_value = (self.img_h - line_bias) / line_slope
        x_bot_value = (0 - line_bias) / line_slope
        y_right_value = self.img_w * line_slope + line_bias
        y_left_value = 0 * line_slope + line_bias

        x_top = floor(x_top_value) if 0 <= x_top_value <= self.img_w else None
        x_bot = floor(x_bot_value) if 0 <= x_bot_value <= self.img_w else None
        y_right = floor(y_right_value) if 0 <= y_right_value <= self.img_h else None
        y_left = floor(y_left_value) if 0 <= y_left_value <= self.img_w else None

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
            render_image[min_y:max_y, min_x:max_x] = node.id
            # cv2.putText(render_image, f"{node.id}", node.bbox["center"], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1,
            #            cv2.LINE_AA)

        # cv2.imshow("test", render_image)
        # cv2.waitKey(0)
        return render_image