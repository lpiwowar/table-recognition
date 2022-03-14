import os
from xml.etree import ElementTree

import cv2
import numpy as np
import torch
from rtree import index
from shapely.geometry import Polygon
from torch_geometric.data import Data

from utils import coords_string_to_tuple_list
from utils import get_multiple_values_from_dict
from utils import range_wrapper


class Graph(object):
    def __init__(self, config, ocr_output_path, ground_truth_path,
                 img_path, edge_discovery_method='k-nearest-neighbors',
                 input_graph_colorer="node-position"):
        Node.NODE_COUNTER = 0

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
                "header_mark": (51, 204, 51),  # Green
                "data": (0, 153, 255),  # Blue
                "data_empty": (255, 255, 255),  # White
                "header_empty": (255, 255, 255),  # White
                "data_mark": (255, 255, 255),  # White
                None: (0, 0, 0)  # Black
            },
            "edge": {
                "horizontal": (204, 0, 0),  # Red
                "vertical": (153, 51, 255),  # Purple
                "cell": (0, 0, 204),  # Red
                "no-relationship": (0, 0, 0)  # Black
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
            cv2.putText(img, f"{node.id}", node.bbox["center"], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                        cv2.LINE_AA)

        # Visualize GT cells
        for node in self.ground_truth_nodes:
            node_coords = node.bbox["corners"]
            cv2.rectangle(img, node_coords[0], node_coords[1], color=colors["node"][node.type], thickness=8)

        img_name = "graph_" + self.img_path.split("/")[-1]
        cv2.imwrite(os.path.join(self.config.visualize_dir, img_name), img)

    def dump(self):
        # Collect node input attributes
        nodes = {}
        for node in self.nodes:
            nodes[node.id] = node.input_feature_vector
        x = torch.tensor([nodes[key] for key in sorted(nodes.keys())])

        # Collect node output attributes
        nodes = {}
        for node in self.nodes:
            nodes[node.id] = node.output_feature_vector
        y = torch.tensor([nodes[key] for key in sorted(nodes.keys())])

        # Collect edges
        nodes1 = [edge.node1.id for edge in self.edges]
        nodes2 = [edge.node2.id for edge in self.edges]
        edge_index = torch.tensor([nodes1, nodes2])

        # Collect edge input and output attributes
        edge_attr = torch.tensor([edge.input_feature_vector for edge in self.edges])
        edge_output_attr = torch.tensor([edge.output_feature_vector for edge in self.edges])

        # Collect positions of nodes in image
        visualize_position = {}
        for node in self.nodes:
            visualize_position[node.id] = node.bbox["center"]
        visualize_position = torch.tensor([visualize_position[key] for key in sorted(visualize_position.keys())])

        # Collect bounding boxes of nodes in image
        node_bounding_box={}
        for node in self.nodes:
            node_bounding_box[node.id] = node.bbox["corners"]
        node_bounding_box = torch.tensor([node_bounding_box[key] for key in sorted(node_bounding_box.keys())])

        data = Data(
                    # --- Input attributes --- #
                    x=x,                                      # Nodes input attributes
                    edge_index=edge_index,                    # Definition of edges
                    edge_attr=edge_attr,                      # Edge input attributes

                    # --- Expected output attributes --- #
                    y=y,                                      # Nodes output attributes
                    edge_output_attr=edge_output_attr,        # Edge output attributes

                    # --- Output of model --- #
                    output_edges=None,
                    output_nodes=None,

                    # --- Auxiliary attributes --- #
                    node_image_position=visualize_position,   # Position of nodes in image
                    node_bounding_box=node_bounding_box,      # Bounding box of node
                    img_path=self.img_path                    # Path to image
                    )
        filename = os.path.basename(self.img_path).split(".")[0]
        torch.save(data, os.path.join(self.config.prepared_data_dir, f'{filename}.pt'))

    def load(self, data):
        pass


class Edge(object):
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        self.type = None

        self.input_feature_vector = None
        self.output_feature_vector = None

    def __eq__(self, other):
        return self.node1.id == other.node1.id and \
               self.node2.id == other.node2.id

    def __hash__(self):
        return hash(self.__repr__())

    def __repr__(self):
        return f"<Edge: node1={self.node1} node2={self.node2}"

    def is_reflexive(self):
        return self.node1 == self.node2


class Node(object):
    NODE_COUNTER = 0

    def __init__(self, polygon_pts):
        self.polygon_pts = polygon_pts
        self.bbox = self.calculate_node_bbox()

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
        for node in self.graph.nodes:
            self.rtree_index_2_node[node.id] = node
            self.rtree_index.insert(node.id, node.bbox["rtree"])


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
    def ocr_node_and_cell_node_overlap(ocr_node: Node, cell_node: Node) -> bool:
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

