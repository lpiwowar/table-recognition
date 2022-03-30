import cv2
import numpy as np

from skimage.draw import line_nd
import matplotlib.pyplot as plt
from typing import Tuple


class NodeVisibility(object):
    def __init__(self, graph):
        self.graph = graph

        img = cv2.imread(self.graph.img_path)
        self.img_h, self.img_w, self.img_c = img.shape

    def discover_edges(self):
        # How to make logical and between the line and rendered boxes image:
        # >>> a = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        # >>> b = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # >>> d = np.dstack([a, b])
        # >>> d.max(axis=2)
        boxes_image = self.render_boxes_image()
        import time
        # for node in self.graph.nodes:
        for x in range(0, 181):
            line_array = self.get_line_array_in_180(x)
            cv2.imshow("test", line_array)
            cv2.waitKey(0)
            # print(line_array)

        pass

    def get_line_array_in_180(self, angle_deg):
        assert 0 <= angle_deg <= 180, "ERROR: get_line_array_in_180() accepts deg value in range <0,180>"

        if 0 <= angle_deg <= 90:
            return self.get_line_array_in_90(angle_deg)
        else:
            line_array = self.get_line_array_in_90(180 - angle_deg)
            return np.flip(line_array, 0)

    def get_line_array_in_90(self, angle_deg):
        assert 0 <= angle_deg <= 90, "ERROR: get_line_array_in_90() accepts deg value in range <0,90>"

        angle_rad = (np.pi / 180) * angle_deg

        center_x = self.img_w // 2
        center_y = self.img_h // 2

        # Angle in (0, 45>
        if angle_deg < 45:
            x_diff = np.floor(np.tan(angle_rad) * center_x)
            start_point = (self.img_h - 1, center_x - x_diff)
            end_point = (0, center_x + x_diff)

        # Angle in <45, 90>
        elif 45 <= angle_deg <= 90:
            y_diff = np.tan((np.pi/2) - angle_rad) * center_y
            start_point = (center_y + y_diff, 0)
            end_point = (center_y - y_diff, self.img_w - 1)

        line_coords = line_nd(start_point, end_point)
        line_array = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
        line_array[line_coords] = 255

        return line_array

    def render_boxes_image(self):
        """Generate 2D-array in which each pixels says id of node it represents"""
        render_image = np.zeros((self.img_h, self.img_w, self.img_c))
        for node in self.graph.nodes:
            (min_x, min_y, max_x, max_y) = node.bbox["rtree"]
            render_image[min_y:max_y, min_x:max_x, :] = 128
            # cv2.putText(render_image, f"{node.id}", node.bbox["center"], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1,
            #            cv2.LINE_AA)

        # cv2.imshow("test", render_image)
        # cv2.waitKey(0)
        return render_image
