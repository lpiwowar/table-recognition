import torch


class Node(object):
    NODE_COUNTER = 0

    def __init__(self, polygon_pts=None):
        self.polygon_pts = polygon_pts
        if self.polygon_pts:
            self.bbox = self.calculate_node_bbox()

        self.x = None
        self.y = None

        self.id = Node.NODE_COUNTER
        Node.NODE_COUNTER += 1

        self.type = None
        self.start_row = None
        self.end_row = None
        self.start_col = None
        self.end_col = None

        self.output_feature_vector = None
        self.input_feature_vector = None

    def __repr__(self):
        return f"<Node: rtree_id={self.id}>"

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.__repr__())

    def calculate_node_bbox(self):
        x_coords = [x for x, _ in self.polygon_pts]
        y_coords = [y for _, y in self.polygon_pts]
        max_x, min_x = max(x_coords), min(x_coords)
        max_y, min_y = max(y_coords), min(y_coords)

        bbox_coord_types = {
            "rtree": (min_x, min_y, max_x, max_y),
            "polygon": [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)],
            "corners": [(min_x, min_y), (max_x, max_y)],
            "center": [int(min_x + ((max_x - min_x) / 2)), int(min_y + ((max_y - min_y) / 2))]
        }

        return bbox_coord_types
