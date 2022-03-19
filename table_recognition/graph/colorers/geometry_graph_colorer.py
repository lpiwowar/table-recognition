import cv2
import numpy as np


class GeometryGraphColorer(object):
    def __init__(self, graph):
        self.graph = graph
        self.img_height, self.img_width, _ = cv2.imread(self.graph.img_path).shape

    def color_graph(self):
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

        pass