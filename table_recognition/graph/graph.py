import os
from xml.etree import ElementTree

import cv2
import torch
from torch_geometric.data import Data

from table_recognition.graph.colorers import BasicGraphColorer
from table_recognition.graph.colorers import GeometryGraphColorer
from table_recognition.graph.colorers import OutputGraphColorer
from table_recognition.graph.utils import coords_string_to_tuple_list
from table_recognition.graph.edge_discovery import KNearestNeighbors
from table_recognition.graph.utils import roi_align_single_image


class Graph(object):
    def __init__(self, config, ocr_output_path, ground_truth_path,
                 img_path, edge_discovery_method='k-nearest-neighbors',
                 input_graph_colorer="basic-graph-colorer"):
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
            "basic-graph-colorer": BasicGraphColorer(self),
            "geometry-graph-colorer": GeometryGraphColorer(self)
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

            # edge_center = [(node1_center[0] + node2_center[0]) // 2, (node1_center[1] + node2_center[1]) // 2]
            # cv2.putText(img, f"{edge.input_feature_vector[-1]:.2f}, {edge.input_feature_vector[-2]:.2f}", edge_center, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1,
            #            cv2.LINE_AA)


        # Visualize nodes
        for node in self.nodes:
            # Visualize text line region
            node_coords = node.bbox["corners"]
            cv2.rectangle(img, node_coords[0], node_coords[1], color=colors["node"][node.type], thickness=2)

            # Visualize node
            cv2.circle(img, node.bbox["center"], radius=10, color=colors["node"][node.type], thickness=-1)
            cv2.putText(img, f"{node.id}", node.bbox["center"], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1,
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
        node_bounding_box = {}
        for node in self.nodes:
            node_bounding_box[node.id] = node.bbox["corners"]
        node_bounding_box = torch.tensor([node_bounding_box[key] for key in sorted(node_bounding_box.keys())])

        if self.config.visual_features:
            # Cut out regions from image
            img = cv2.imread(self.img_path)
            img_h, img_w, img_c = img.shape
            node_cut_padding = 20
            node_image_regions = {}
            for node in self.nodes:
                (min_x, min_y, max_x, max_y) = node.bbox["rtree"]

                min_x = max(min_x - node_cut_padding, 0)
                max_x = min(max_x + node_cut_padding, img_w)
                min_y = max(min_y - node_cut_padding, 0)
                max_y = min(max_y + node_cut_padding, img_h)

                img_region = img[min_y:max_y, min_x:max_x, :]
                node_image_regions[node.id] = torch.tensor(img_region)
            node_image_regions = [roi_align_single_image(node_image_regions[key], (10, 10))
                                  for key in sorted(node_image_regions.keys())]
            print(self.img_path)
            node_image_regions = torch.stack(node_image_regions)

            edge_cut_padding = 20
            edge_image_regions = []
            for edge in self.edges:
                (node1_min_x, node1_min_y, node1_max_x, node1_max_y) = edge.node1.bbox["rtree"]
                (node2_min_x, node2_min_y, node2_max_x, node2_max_y) = edge.node2.bbox["rtree"]

                # Find bounding box of the two bounding boxes
                min_x = min(node1_min_x, node2_min_x)
                max_x = max(node1_max_x, node2_max_x)
                min_y = min(node1_min_y, node2_min_y)
                max_y = max(node1_max_y, node2_max_y)

                # Add padding
                min_x = max(min_x - edge_cut_padding, 0)
                max_x = min(max_x + edge_cut_padding, img_w)
                min_y = max(min_y - edge_cut_padding, 0)
                max_y = min(max_y + edge_cut_padding, img_h)

                img_region = img[min_y:max_y, min_x:max_x, :]
                edge_image_regions += [roi_align_single_image(torch.tensor(img_region), (16, 16))]
            edge_image_regions = torch.stack(edge_image_regions)

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
                    img_path=self.img_path,                   # Path to image
                    node_image_regions=node_image_regions,    # Cutted out regions from images (nodes)
                    edge_image_regions=edge_image_regions     # Cutted out regions from images (edges)
                    )

        filename = os.path.basename(self.img_path).split(".")[0]
        torch.save(data, os.path.join(self.config.prepared_data_dir, f'{filename}.pt'))

    def load(self, data):
        pass


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




