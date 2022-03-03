import os
from xml.etree import ElementTree

import cv2
from rtree import index
from shapely.geometry import Polygon
# from torch_geometric.data import Data

from utils import coords_string_to_tuple_list
from utils import get_multiple_values_from_dict
from utils import range_wrapper


class Graph(object):
    def __init__(self, config, ocr_output_path, ground_truth_path,
                 img_path, edge_discovery_method='k-nearest-neighbors',
                 input_graph_colorer="node-position"):
        self.config = config
        self.ocr_output_path = ocr_output_path
        self.ground_truth_path = ground_truth_path
        self.img_path = img_path
        self.edge_discovery_method = edge_discovery_method
        self.input_graph_colorer = input_graph_colorer

        self.edges = set()
        self.nodes = set()

        self.edge_discovery_methods = {
            "k-nearest-neighbors": KNearestNeighbors(self)
        }

        self.ground_truth_nodes = []
        self.output_graph_colorer = OutputGraphColorer(self)

        self.input_graph_colorers = {
            "node-position": InputGraphColorerNodePosition(self)
        }

    def initialize(self):
        ns = {"xmlns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15",
              "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
              "xsi:schemaLocation": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd"}

        xpath_coords = "./xmlns:Page/xmlns:TextRegion/xmlns:TextLine"

        # Parse OCR output
        text_lines_xml = ElementTree.parse(self.ocr_output_path).findall(xpath_coords, ns)
        for text_line_xml in text_lines_xml:
            coords_xml = text_line_xml.find("./xmlns:Coords", ns)
            polygon_pts = coords_string_to_tuple_list(coords_xml.attrib["points"])
            self.nodes.add(Node(polygon_pts))

        # Discover edges
        self.edge_discovery_methods[self.edge_discovery_method].discover_edges()

    def color_output(self):
        # Parse XML ground truth file
        cells_xml = ElementTree.parse(self.ground_truth_path).findall("./table/cell")
        for cell_xml in cells_xml:
            coords_xml = cell_xml.find("./Coords")
            polygon_pts = coords_string_to_tuple_list(coords_xml.attrib["points"])
            node = Node(polygon_pts)
            node.type = cell_xml.attrib["type"]
            node.start_row = int(cell_xml.attrib["start-row"])
            node.end_row = int(cell_xml.attrib["end-row"])
            node.start_col = int(cell_xml.attrib["start-col"])
            node.end_col = int(cell_xml.attrib["end-col"])
            self.ground_truth_nodes += [node]

        # Color graph
        self.output_graph_colorer.color_graph()

    def color_input(self):
        self.input_graph_colorers[self.input_graph_colorer].color_graph()

    def visualize(self):
        colors = {
            "node": {
                "header": (51, 204, 51),  # Green
                "data": (0, 153, 255),  # Blue
                "data_empty": (255, 255, 255),  # White
                "header_empty": (255, 255, 255),  # White
                "data_mark": (255, 255, 255),  # White
                None: (0, 0, 0)  # Black
            },
            "edge": {
                "horizontal": (204, 0, 0),  # Red
                "vertical": (153, 51, 255),  # Purple
                "cell": (0, 0, 204), # Red
                None: (0, 0, 0)  # Black
            }
        }

        img = cv2.imread(self.img_path)

        # Visualize edges
        for edge in self.edges:
            node1_center = edge.node1.bbox["center"]
            node2_center = edge.node2.bbox["center"]
            cv2.line(img, node1_center, node2_center, color=colors["edge"][edge.type], thickness=3)

        # Visualize nodes
        for node in self.nodes:
            # Visualize text line region
            node_coords = node.bbox["corners"]
            cv2.rectangle(img, node_coords[0], node_coords[1], color=colors["node"][node.type], thickness=2)

            # Visualize node
            cv2.circle(img, node.bbox["center"], radius=10, color=colors["node"][node.type], thickness=-1)

        # Visualize GT cells
        for node in self.ground_truth_nodes:
            node_coords = node.bbox["corners"]
            cv2.rectangle(img, node_coords[0], node_coords[1], color=colors["node"][node.type], thickness=8)

        img_name = "graph_" + self.img_path.split("/")[-1]
        cv2.imwrite(os.path.join(self.config.visualize_dir, img_name), img)

    def dump(self):
        x = [node.feature_vector for node in self.nodes]
        


class Edge(object):
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        self.type = None

    def __eq__(self, other):
        return self.node1.rtree_id == other.node1.rtree_id and \
               self.node2.rtree_id == other.node2.rtree_id

    def __hash__(self):
        return hash(self.__repr__())

    def __repr__(self):
        return f"<Edge: node1={self.node1} node2={self.node2}"

    def is_reflexive(self):
        return self.node1 == self.node2


class Node(object):
    def __init__(self, polygon_pts):
        self.polygon_pts = polygon_pts
        self.bbox = self.get_node_bbox()

        self.rtree_id = None
        self.type = None

        self.start_row = None
        self.end_row = None
        self.start_col = None
        self.end_col = None

        self.feature_vector = None

    def __repr__(self):
        return f"<Node: rtree_id={self.rtree_id}>"

    def get_node_bbox(self):
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


class KNearestNeighbors(object):
    K_NEIGHBORS = 4

    def __init__(self, graph):
        self.graph = graph

        self.rtree_index = index.Index()
        self.rtree_index_2_node = {}

    def discover_edges(self):
        self.populate_rtree_index()

        # Discover K nearest neighbors for each node
        for node in self.graph.nodes:
            node_neighbors_idx = list(self.rtree_index.nearest(node.bbox["rtree"],
                                                               KNearestNeighbors.K_NEIGHBORS))
            self.graph.edges = self.graph.edges.union({Edge(node, self.rtree_index_2_node[neighbor_idx])
                                                       for neighbor_idx in node_neighbors_idx})
            self.graph.edges = self.graph.edges.union({Edge(self.rtree_index_2_node[neighbor_idx], node)
                                                       for neighbor_idx in node_neighbors_idx})

        # Remove reflexive edges
        self.graph.edges = {edge for edge in self.graph.edges if not edge.is_reflexive()}

    def populate_rtree_index(self):
        rtree_id = 0
        for node in self.graph.nodes:
            node.rtree_id = rtree_id
            self.rtree_index_2_node[node.rtree_id] = node
            rtree_id += 1
            self.rtree_index.insert(node.rtree_id, node.bbox["rtree"])


class OutputGraphColorer(object):
    def __init__(self, graph):
        self.graph = graph

        self.rtree_index = index.Index()
        self.rtree_index_2_node = {}

    def color_graph(self):
        self.populate_rtree_index()
        self.color_nodes()
        self.color_edges()

    def populate_rtree_index(self):
        rtree_id = 0
        for node in self.graph.ground_truth_nodes:
            node.rtree_id = rtree_id
            self.rtree_index_2_node[node.rtree_id] = node
            rtree_id += 1
            self.rtree_index.insert(node.rtree_id, node.bbox["rtree"])

    def color_nodes(self):
        for node in self.graph.nodes:
            cells_intersections_idx = list(self.rtree_index.intersection(node.bbox["rtree"]))
            if cells_intersections_idx:
                cells_intersections_nodes = get_multiple_values_from_dict(self.rtree_index_2_node,
                                                                          cells_intersections_idx)
                cells_intersections_types = [node.type for node in cells_intersections_nodes]
                cell_type = OutputGraphColorer.majority_type(cells_intersections_types)
                node.type = cell_type

    def color_edges(self):
        for edge in self.graph.edges:
            node1_logical_position = self.get_logical_position(edge.node1)
            node2_logical_position = self.get_logical_position(edge.node2)
            edge.type = OutputGraphColorer.get_edge_type(node1_logical_position, node2_logical_position)

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
    def ocr_node_and_cell_node_overlap(ocr_node: Node, cell_node: Node) -> bool:
        ocr_polygon = Polygon(ocr_node.bbox["polygon"])
        cell_polygon = Polygon(cell_node.bbox["polygon"])
        intersection = ocr_polygon.intersection(cell_polygon).area
        if (intersection / ocr_polygon.area) > 0.9:
            return True
        elif (intersection / cell_polygon.area) > 0.1:
            return True
        else:
            return False

    @staticmethod
    def get_edge_type(node1, node2):
        if None in node1.values() or None in node2.values():
            return None

        node1_row_range = set(range_wrapper(node1["start-row"], node1["end-row"]))
        node1_col_range = set(range_wrapper(node1["start-col"], node1["end-col"]))
        node2_row_range = set(range_wrapper(node2["start-row"], node2["end-row"]))
        node2_col_range = set(range_wrapper(node2["start-col"], node2["end-col"]))

        if node1_row_range == node2_row_range and node1_col_range == node2_col_range:
            return "cell"

        if node1_row_range <= node2_row_range or node2_row_range <= node1_row_range:
            return "vertical"

        if node1_col_range <= node2_col_range or node2_col_range <= node1_col_range:
            return "horizontal"

        return None

    @staticmethod
    def majority_type(types):
        types_order = {"data": 1, "header": 1, "data_empty": 0, "header_empty": 0,
                       "data_mark": 0}
        return max(types, key=lambda x: types_order[x])


class InputGraphColorerNodePosition(object):
    def __init__(self, graph):
        self.graph = graph

    def color_graph(self):
        for node in self.graph.nodes:
            img = cv2.imread(self.graph.img_path)
            x, y = node.bbox["center"]
            position = [x / img.width, y / img.height]
            node.feature_vector = position
