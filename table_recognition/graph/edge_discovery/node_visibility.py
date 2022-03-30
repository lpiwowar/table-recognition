import cv2
import numpy as np

from skimage.draw import line_nd


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
        import time
        print(len(self.graph.nodes))
        start = time.time()
        print("hello")
        for x in range(0, 181):
            line_array = self.get_line((210, 148), x)
            # line_array[(148-10):(148+10), (210-10):(210+10)] = 255
            # cv2.imshow("test", line_array)
            # cv2.waitKey(0)
        end = time.time()
        print(end - start)

    def get_line(self, point, angle_deg):
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
            return line_array

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

        x_top = x_top_value if 0 <= x_top_value <= self.img_w else 0
        x_bot = x_bot_value if 0 <= x_bot_value <= self.img_w else 0
        y_right = y_right_value if 0 <= y_right_value <= self.img_h else 0
        y_left = y_left_value if 0 <= y_left_value <= self.img_w else 0

        # Find the two points
        line_points = []
        line_points += [(self.img_h - 1, x_top)] if x_top else []
        line_points += [(0, x_bot)] if x_bot else []
        line_points += [(y_right, self.img_w - 1)] if y_right else []
        line_points += [(y_left, 0)] if y_left else []

        line_coords = line_nd(line_points[0], line_points[1])
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
