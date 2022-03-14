from rtree import index

from shapely.geometry import Polygon

from table_recognition.graph.colorers.utils import get_multiple_values_from_dict
from table_recognition.graph.colorers.utils import range_wrapper


class OutputGraphColorer(object):
    TYPE_2_FEATURE_VECTOR = {
       "node": {
           "header": [1, 0],
           "header_mark": [1, 0],
           "header_empty": [1, 0],
           "data": [0, 1],
           "data_mark": [0, 1],
           "data_empty": [0, 1],
           None: [0, 1]
       },
       "edge": {
           "cell": [1, 0, 0, 0],
           "horizontal": [0, 1, 0, 0],
           "vertical": [0, 0, 1, 0],
           "no-relationship": [0, 0, 0, 1]
       }
    }

    def __init__(self, graph):
        self.graph = graph

        self.rtree_index = index.Index()
        self.rtree_index_2_node = {}

    def color_graph(self):
        self.populate_rtree_index()
        self.color_nodes()
        self.color_edges()

    def populate_rtree_index(self):
        for node in self.graph.ground_truth_nodes:
            self.rtree_index_2_node[node.id] = node
            self.rtree_index.insert(node.id, node.bbox["rtree"])

    def color_nodes(self):
        for node in self.graph.nodes:
            cells_intersections_idx = list(self.rtree_index.intersection(node.bbox["rtree"]))
            if cells_intersections_idx:
                cells_intersections_nodes = get_multiple_values_from_dict(self.rtree_index_2_node,
                                                                          cells_intersections_idx)
                cells_intersections_types = [node.type for node in cells_intersections_nodes]
                cell_type = OutputGraphColorer.majority_type(cells_intersections_types)
                node.type = cell_type
                node.output_feature_vector = OutputGraphColorer.TYPE_2_FEATURE_VECTOR["node"][node.type]
            else:
                # WARNING
                node.output_feature_vector = [0, 1]

    def color_edges(self):
        for edge in self.graph.edges:
            node1_logical_position = self.get_logical_position(edge.node1)
            node2_logical_position = self.get_logical_position(edge.node2)
            edge.type = OutputGraphColorer.get_edge_type(node1_logical_position, node2_logical_position)
            edge.output_feature_vector = OutputGraphColorer.TYPE_2_FEATURE_VECTOR["edge"][edge.type]

    def get_logical_position(self, node):
        nodes_position = {"start-row": None, "end-row": None,
                          "start-col": None, "end-col": None}
        cells_intersections_idx = list(self.rtree_index.intersection(node.bbox["rtree"]))
        if not cells_intersections_idx:
            return nodes_position

        cells_intersections_nodes = [self.rtree_index_2_node[cell_idx]
                                     for cell_idx in cells_intersections_idx]

        # Filter out cells that have small intersection with the text line
        cells_intersections_nodes = [cell_node for cell_node in cells_intersections_nodes
                                     if OutputGraphColorer.ocr_node_and_cell_node_overlap(node, cell_node)]

        if not cells_intersections_nodes:
            return nodes_position

        nodes_position["start-row"] = min([node.start_row for node in cells_intersections_nodes])
        nodes_position["end-row"] = max([node.end_row for node in cells_intersections_nodes])
        nodes_position["start-col"] = min([node.start_col for node in cells_intersections_nodes])
        nodes_position["end-col"] = min([node.end_col for node in cells_intersections_nodes])

        return nodes_position

    @staticmethod
    def ocr_node_and_cell_node_overlap(ocr_node, cell_node) -> bool:
        ocr_polygon = Polygon(ocr_node.bbox["polygon"])
        cell_polygon = Polygon(cell_node.bbox["polygon"])
        intersection = ocr_polygon.intersection(cell_polygon).area
        if (intersection / max(10e-10, ocr_polygon.area)) > 0.9:
            return True
        elif (intersection / max(10e-10, cell_polygon.area)) > 0.1:
            return True
        else:
            return False

    @staticmethod
    def get_edge_type(node1, node2):
        if None in node1.values() or None in node2.values():
            return "no-relationship"

        node1_row_range = set(range_wrapper(node1["start-row"], node1["end-row"]))
        node1_col_range = set(range_wrapper(node1["start-col"], node1["end-col"]))
        node2_row_range = set(range_wrapper(node2["start-row"], node2["end-row"]))
        node2_col_range = set(range_wrapper(node2["start-col"], node2["end-col"]))

        if node1_row_range == node2_row_range and node1_col_range == node2_col_range:
            return "cell"

        # Nodes are in the same column
        if node1_row_range <= node2_row_range or node2_row_range <= node1_row_range:
            if OutputGraphColorer.nodes_vertically_visible(node1, node2):
                return "vertical"
            else:
                return "no-relationship"

        # Nodes are in the same row
        if node1_col_range <= node2_col_range or node2_col_range <= node1_col_range:
            if OutputGraphColorer.nodes_horizontally_visible(node1, node2):
                return "horizontal"
            else:
                return "no-relationship"

        return "no-relationship"

    @staticmethod
    def nodes_vertically_visible(node1, node2):
        bottom_cell_row = max(node1["start-row"], node2["start-row"])
        top_cell_row = min(node1["end-row"], node2["end-row"])
        return True if abs(bottom_cell_row - top_cell_row) <= 1 else False

    @staticmethod
    def nodes_horizontally_visible(node1, node2):
        right_cell_column = max(node1["start-col"], node2["start-col"])
        left_cell_column = min(node1["end-col"], node2["end-col"])
        return True if abs(right_cell_column - left_cell_column) <= 1 else False

    @staticmethod
    def majority_type(types):
        types_order = {"data": 1, "header": 1, "header_empty": 1, "header_mark": 1, "data_empty": 0, "header_empty": 0,
                       "data_mark": 0}
        return max(types, key=lambda x: types_order[x])

