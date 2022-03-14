import cv2
import numpy as np


class InputGraphColorerNodePosition(object):
    def __init__(self, graph):
        self.graph = graph
        self.img_height, self.img_width, _ = cv2.imread(self.graph.img_path).shape

    def color_graph(self):
        self.color_nodes()
        self.color_edges()

    def color_nodes(self):
        for node in self.graph.nodes:
            x, y = node.bbox["center"]
            position = [x / self.img_width, y / self.img_height]

            [(min_x, min_y), (max_x, max_y)] = node.bbox["corners"]
            bbox_width = abs(max_x - min_x) / self.img_width
            bbox_height = abs(max_y - min_y) / self.img_height

            node.input_feature_vector = position + [bbox_width, bbox_height]

    def color_edges(self):
        for edge in self.graph.edges:
            # Center of node1
            node1_x, node1_y = edge.node1.bbox["center"]
            node1_x, node1_y = node1_x / self.img_width, node1_y / self.img_height

            # Center of node2
            node2_x, node2_y = edge.node2.bbox["center"]
            node2_x, node2_y = node2_x / self.img_width, node2_y / self.img_height

            # Feature1: Distance of the two centers
            distance = np.linalg.norm(np.array([node1_x, node1_y]) - np.array([node2_x, node2_y]))

            # Feature2: Average of the two centers
            avg_position_x, avg_position_y = (node1_x + node2_x) / 2, (node1_y + node2_y) / 2

            edge.input_feature_vector = [float(distance)] + [avg_position_x, avg_position_y]

