import cv2
import numpy as np

from skimage.draw import line_nd

from table_recognition.graph import Graph


class NodeVisibility(object):
    def __init__(self, graph: Graph):
        self.graph = graph

    def discover_edges(self):
        # How to make logical and between the line and rendered boxes image:
        # >>> a = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        # >>> b = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # >>> d = np.dstack([a, b])
        # >>> d.max(axis=2)
        pass

    def render_boxes_image(self):
        """Generate 2D-array in which each pixels says id of node it represents"""
        img = cv2.imread(self.graph.img_path)
        img_h, img_w, img_c = img.shape

        render_image = np.zeros((img_h, img_w))
        for node in self.graph.nodes:
            (min_x, min_y, max_x, max_y) = node.bbox["rtree"]
            render_image[min_y:max_y, min_x:max_x] = node.id

        return render_image
