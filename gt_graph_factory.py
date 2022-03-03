import os
from operator import itemgetter
from xml.etree import ElementTree

import cv2
from rtree import index
from shapely.geometry import Polygon

from utils import coords_string_to_tuple_list


# noinspection PyUnresolvedReferences
class GTGraphCreator(object):
    """
    Class that creates ground truth graph representation.

    This class uses output from the OCR and XML CTDAR ground truth
    data to create the graph representation of the expected output
    of the GNN.

    :type ocr_file_path:    String
    :param ocr_file_path:   Absolute path to XML file containing the OCR output
    :type dataset_gt_path:    String
    :param dataset_gt_path:   Absolute path to XML file containing the dataset GT
    :type dataset_image_path: String
    :type dataset_image_path: Absolute path to image from the dataset
    """

    def __init__(self, ocr_file_path, dataset_gt_path, dataset_image_path=None):
        self.ocr_file_path = ocr_file_path
        self.dataset_gt_path = dataset_gt_path
        self.dataset_image_path = dataset_image_path
        self.text_lines = []                         # List of OCRTextLine objects
        self.rtree_id_to_text_line = {}              # Directory that maps rtree ID to OCRTextLine instance
        self.edges = set()                           # List of edges between the textlines [[1,2], [2,3], [4,5], ...]

        self.table_cells = []                        # List of table cells from the ground truth
        self.d_rtree_id_to_text_line = {}            # Directory that maps rtree ID to DatasetGTNode

    def create_k_nearest_neighbors_graphs(self, k_param=4):
        """
        Create graph representation of XML CTDAR ground truth data.
        The created graph representation can be used to train the GNN.
        """

        ns = {"xmlns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15",
              "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
              "xsi:schemaLocation": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd"}

        xpath_coords = "./xmlns:Page/xmlns:TextRegion/xmlns:TextLine"

        # Parse XML OCR output
        coords_xml = ElementTree.parse(self.ocr_file_path).findall(xpath_coords, ns)
        self.text_lines += [OCRTextLine(coord_xml) for coord_xml in coords_xml]

        # Create Rtree index
        idx = index.Index()
        rtree_id = 0
        for text_line in self.text_lines:
            text_line.rtree_id = rtree_id
            self.rtree_id_to_text_line[text_line.rtree_id] = text_line
            rtree_id += 1

            idx.insert(text_line.rtree_id, text_line.bbox_points)

        # Parse XML from the dataset
        cells_xml = ElementTree.parse(self.dataset_gt_path).findall("./table/cell")
        self.table_cells += [DatasetGTNode(cell_xml) for cell_xml in cells_xml]

        # Create Rtree index - dataset GT
        idx_cells = index.Index()
        rtree_id = 0
        for table_cell in self.table_cells:
            table_cell.rtree_id = rtree_id
            self.d_rtree_id_to_text_line[table_cell.rtree_id] = table_cell
            rtree_id += 1

            idx_cells.insert(table_cell.rtree_id, table_cell.bbox_points)

        # Create graph representation
        for text_line in self.text_lines:
            # Create edges and define its types
            k_neighbors = list(idx.nearest(text_line.bbox_points, k_param))
            self.edges = self.edges.union({Edge(text_line, self.rtree_id_to_text_line[neighbor])
                                           for neighbor in k_neighbors})
            self.edges = self.edges.union({Edge(self.rtree_id_to_text_line[neighbor], text_line)
                                           for neighbor in k_neighbors})

            # Color type of node - Define type of node in graph
            cells_intersection = list(idx_cells.intersection(text_line.bbox_points))
            if cells_intersection:
                dataset_gt_nodes = self.get_dataset_gt_nodes_by_rtree_idxs(cells_intersection)
                cells_intersection_types = [cell.type for cell in dataset_gt_nodes]
                cell_type = max(cells_intersection_types, key=cells_intersection_types.count)
                text_line.type = cell_type

        # Color edges in graph - define its type
        for edge in self.edges:
            # Get left node intersection with GT dataset
            left_node_intersections_idx = list(idx_cells.intersection(edge.node_left.bbox_points))
            if not left_node_intersections_idx:
                continue

            #if edge.node_left.rtree_id != 649 and edge.node_left.rtree_id != 649:
            #   continue

            left_node_pts = edge.node_left.bbox_polygon
            left_node_intersections = self.get_dataset_gt_nodes_by_rtree_idxs(left_node_intersections_idx)
            # print(f"NODE ID: {edge.node_left.rtree_id} = {left_node_intersections}")
            # print(f"node_left_bbox_points: {edge.node_left.bbox_points}")
            left_node_intersections = [node for node in left_node_intersections
                                       if GTGraphCreator.polygon_eats_polygon_percent(left_node_pts, node.bbox_polygon) > 0.2]
            # print(f"left_filtered: {left_node_intersections}")

            if not left_node_intersections:
                continue

            left_node_start_row = min([node.start_row for node in left_node_intersections])
            left_node_end_row = max([node.end_row for node in left_node_intersections])
            left_node_start_col = min([node.start_col for node in left_node_intersections])
            left_node_end_col = max([node.end_col for node in left_node_intersections])

            edge.node_left.max_start_row = left_node_start_row
            edge.node_left.max_end_row = left_node_end_row
            edge.node_left.max_start_col = left_node_start_col
            edge.node_left.max_end_col = left_node_end_col

            # Get right node intersection with GT dataset
            right_node_intersections_idx = list(idx_cells.intersection(edge.node_right.bbox_points))
            if not right_node_intersections_idx:
                continue

            right_node_pts = edge.node_right.bbox_polygon
            right_node_intersections = self.get_dataset_gt_nodes_by_rtree_idxs(right_node_intersections_idx)
            # print(f"NODE ID: {edge.node_right.rtree_id} = {right_node_intersections}")
            # print(f"node_right_bbox_points: {edge.node_right.bbox_points}")
            # print(f"right_filtered: {right_node_intersections}")
            right_node_intersections = [node for node in right_node_intersections
                                        if GTGraphCreator.polygon_eats_polygon_percent(right_node_pts, node.bbox_polygon) > 0.2]

            if not right_node_intersections:
                continue

            right_node_start_row = min([node.start_row for node in right_node_intersections])
            right_node_end_row = max([node.end_row for node in right_node_intersections])
            right_node_start_col = min([node.start_col for node in right_node_intersections])
            right_node_end_col = max([node.end_col for node in right_node_intersections])

            edge.node_right.max_start_row = right_node_start_row
            edge.node_right.max_end_row = right_node_end_row
            edge.node_right.max_start_col = right_node_start_col
            edge.node_right.max_end_col = right_node_end_col

            edge.type = GTGraphCreator.get_edge_type((left_node_start_row, left_node_end_row,
                                                      left_node_start_col, left_node_end_col),
                                                     (right_node_start_row, right_node_end_row,
                                                      right_node_start_col, right_node_end_col))

    @staticmethod
    def polygon_eats_polygon_percent(polygon_1, polygon_2):
        #print(f"polygon_1: {polygon_1} polygon_2: {polygon_2}")
        #intersection_area = Polygon(polygon_1).intersection(Polygon(polygon_2)).area
        #polygon_1_area = Polygon(polygon_1).area
        #if polygon_1_area == 0:
        #    return 0
        #else:
            #print(intersection_area / polygon_2_area)
            #print(f"intersection_area: {intersection_area} polygon_2_area: {polygon_2_area} => {intersection_area/ polygon_2_area}")
        #    return intersection_area / Polygon(polygon_1).union(Polygon(polygon_2)).area

        polygon_1 = Polygon(polygon_1)
        polygon_2 = Polygon(polygon_2)
        intersection = polygon_1.intersection(polygon_2).area
        if intersection / polygon_1.area > 0.9:
            return 1
        elif intersection / polygon_2.area > 0.1:
            return 1
        else:
            return 0

    @staticmethod
    def get_edge_type(left_node_table_position, right_node_table_position):
        def my_range(start, end):
            return list(range(start, end)) + [end]

        l_start_row, l_end_row, l_start_col, l_end_col = left_node_table_position
        r_start_row, r_end_row, r_start_col, r_end_col = right_node_table_position

        if set(my_range(l_start_row, l_end_row)) <= set(my_range(r_start_row, r_end_row)) or \
           set(my_range(r_start_row, r_end_row)) <= set(my_range(l_start_row, l_end_row)):
            return "vertical"

        if set(my_range(l_start_col, l_end_col)) <= set(my_range(r_start_col, r_end_col)) or \
           set(my_range(r_start_col, r_end_col)) <= set(my_range(l_start_col, l_end_col)):
            return "horizontal"

        return None

    def get_dataset_gt_nodes_by_rtree_idxs(self, rtree_idxs):
        """
        A function that takes as an input a list of rtree_idxs and
        returns a list of corresponding dataset_gt_objects

        :type rtree_idxs:  List of integers
        :param rtree_idxs: List of rtree indices
        :return:           List of DatasetGTNode objects
        """
        if not rtree_idxs:
            return []

        dataset_gt_nodes = itemgetter(*rtree_idxs)(self.d_rtree_id_to_text_line)

        if type(dataset_gt_nodes) is tuple:
            return list(dataset_gt_nodes)
        else:
            return [dataset_gt_nodes]

    def visualize_graph(self, visualize_dir="./dataset_preparation/visualization"):
        """
        Visualize the created graph. This function MUST be called after one of the
        functions for graph creation create_*

        :type visualize_dir:  String
        :param visualize_dir: Directory that should store the visualizations
        """
        colors = {
            "node": {
                "header": (0, 255, 0),
                "data": (255, 0, 0),
                "data_empty": (230, 230, 250),
                "header_empty": (255, 248, 220),
                "data_mark": (0, 0, 0),
                None: (0, 0, 0)
            },
            "edge": {
                "horizontal": (255, 0, 0),
                "vertical": (255, 255, 0),
                None: (0, 0, 0)
            }
        }

        img = cv2.imread(self.dataset_image_path)

        for text_line in self.text_lines:
            # Draw polygon that defines region for a given text line
            left, bottom, right, top = text_line.bbox_points
            cv2.rectangle(img, (left, bottom), (right, top), color=(0, 0, 255), thickness=2)

            # Draw edges for a given node
            neighbors = [edge for edge in self.edges if edge.node_left.rtree_id == text_line.rtree_id]
            for neighbor in neighbors:
                src_pt = self.rtree_id_to_text_line[text_line.rtree_id].bbox_center()
                dst_pt = self.rtree_id_to_text_line[neighbor.node_right.rtree_id].bbox_center()
                cv2.line(img, src_pt, dst_pt, color=colors["edge"][neighbor.type], thickness=3)

            # Draw graphs node (circle in the middle of the text line)
            center_x, center_y = text_line.bbox_center()
            cv2.circle(img, (center_x, center_y), radius=10, color=colors["node"][text_line.type], thickness=-1)

        for text_line in self.text_lines:
            center_x, center_y = text_line.bbox_center()
            cv2.putText(img,
                      #  f"{text_line.max_start_row},{text_line.max_end_row},{text_line.max_start_col},{text_line.max_end_col}",
                        f"{text_line.rtree_id}",
                        (center_x - 10, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        for table_cell in self.table_cells:
            left, bottom, right, top = table_cell.bbox_points
            cv2.putText(img, f"{table_cell.rtree_id}", (left + 10, top + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (255, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(img, (left, bottom), (right, top), color=(255, 0, 0), thickness=1)

        image_name = self.dataset_image_path.split("/")[-1]
        cv2.imwrite(os.path.join(visualize_dir, "GRAPH_" + image_name), img)


class Edge(object):
    def __init__(self, node_left, node_right):
        self.node_left = node_left
        self.node_right = node_right
        self.type = None

    def __eq__(self, other):
        return str(self.node_left.rtree_id) == str(other.node_left.rtree_id) and \
               str(self.node_right.rtree_id) == str(other.node_right.rtree_id)

    def __hash__(self):
        return hash(self.__repr__())

    def __repr__(self):
        return f"[{self.node_left.rtree_id}, {self.node_right.rtree_id}]"


class PolygonNode(object):
    """
    This class represents a region in an image that is defined
    by a polygon.

    The region is defined by XML tag. And the properties of the
    region are used to define a node in a graph that is used
    to train the GNN.

    :type text_line_xml:  XMLElement
    :param text_line_xml: XML element representing coordinates of a given text line
    """
    def __init__(self, text_line_xml):
        """
        A constructor of PolygonNode class

        :type text_line_xml:  XMLElement
        :param text_line_xml: XML element representing coordinates of a given text line
        """

        # REMOVE LATER!!!
        self.max_start_row, self.max_end_row, self.max_start_col, self.max_end_col = None, None, None, None

        self.text_line_xml = text_line_xml
        self.polygon_points = None
        self.bbox_points = None
        self.bbox_polygon = None
        self.rtree_id = None

    def __repr__(self):
        return str(self.rtree_id)

    def parse(self):
        """Parser of the information from the :param text_line_xml:"""
        raise Exception(f"parse() not implemented!")

    def calculate_bbox_points(self):
        """
        Calculate a bounding box of a given polygon specified
        in the :param polygon_points: variable.
        """

        x_coords = [x for x, _ in self.polygon_points]
        y_coords = [y for _, y in self.polygon_points]
        max_x, min_x = max(x_coords), min(x_coords)
        max_y, min_y = max(y_coords), min(y_coords)
        self.bbox_points = (min_x, min_y, max_x, max_y)
        self.bbox_polygon = [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]

    def bbox_center(self):
        """
        Returns 2D point that is in the middle of the bounding box
        defined by polygon points

        :return: tuple that contains coordinates of 2D point that is in
                 the middle of the bounding box
        """
        left, bottom, right, top = self.bbox_points
        width = abs(right - left)
        height = abs(top - bottom)
        center_x = int(left + width / 2)
        center_y = int(top - height / 2)

        return center_x, center_y


class OCRTextLine(PolygonNode):
    """
    Class representing a given text line.

    The class takes as an input XML element representing given text line.
    The XML element is parsed by the class and attributes of the element
    are stored inside the object.

    :type text_line_xml:  XMLElement
    :param text_line_xml: XML element representing coordinates of a given text line
    """

    def __init__(self, text_line_xml):
        super().__init__(text_line_xml)

        self.uuid = None  # UUID that represents the text line in XML
        self.type = None
        self.parse()
        self.calculate_bbox_points()

    def parse(self):
        """Parser that parser information from the :param text_line_xml:"""
        ns = {"xmlns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15",
              "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
              "xsi:schemaLocation": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd"}

        self.uuid = self.text_line_xml.attrib["id"]

        coords_xml = self.text_line_xml.find("./xmlns:Coords", ns)
        self.polygon_points = coords_string_to_tuple_list(coords_xml.attrib["points"])


class DatasetGTNode(PolygonNode):
    """
    Class representing a polygon that defines a table cell in the dataset.

    The class takes as an input XML element representing given table cell
    The XML element is parsed by the class and attributes of the element
    are stored inside the object.

    :type text_line_xml:  XMLElement
    :param text_line_xml: XML element representing coordinates of a given text line
    """

    def __init__(self, text_line_xml):
        super().__init__(text_line_xml)

        self.start_row = None
        self.end_row = None
        self.start_col = None
        self.end_col = None
        self.type = None

        self.parse()
        self.calculate_bbox_points()

    def parse(self):
        """Parser that parser information from the :param text_line_xml:"""
        self.start_row = int(self.text_line_xml.attrib["start-row"])
        self.end_row = int(self.text_line_xml.attrib["end-row"])
        self.start_col = int(self.text_line_xml.attrib["start-col"])
        self.end_col = int(self.text_line_xml.attrib["end-col"])
        self.type = str(self.text_line_xml.attrib["type"])

        coords_xml = self.text_line_xml.find("./Coords")
        self.polygon_points = coords_string_to_tuple_list(coords_xml.attrib["points"])

"""
    def __init__(self, text_line_xml):
        ""
        A constructor for OCRNode class

        :param text_line_xml: XML element representing a given text line

        ""
        self.text_line_xml = text_line_xml  # XML representation of TextLine
        self.uuid = None  # UUID that represents the text line in XML
        self.rtree_id = None  # ID that represents the polygon in rtree db
        self.polygon_points = []  # List of 2D coords that define the text line polygon
        self.bbox_points = None  # The minimal bounding box (left, bottom, right, top)

        self.parse()

    def __str__(self):
        return f"<OCRNode {self.polygon_points=}>"

    def parse(self):
        ns = {"xmlns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15",
              "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
              "xsi:schemaLocation": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd"}

        self.uuid = self.text_line_xml.attrib["id"]

        coords_xml = self.text_line_xml.find("./xmlns:Coords", ns)
        self.polygon_points = coords_string_to_tuple_list(coords_xml.attrib["points"])
        self.calculate_bbox_points()
        
    def calculate_bbox_points(self):
        ""
        Calculate a bounding box of a given polygon specified
        in the :param polygon_points: variable.
        ""

        x_coords = [x for x, _ in self.polygon_points]
        y_coords = [y for _, y in self.polygon_points]
        max_x, min_x = max(x_coords), min(x_coords)
        max_y, min_y = max(y_coords), min(y_coords)
        self.bbox_points = (min_x, min_y, max_x, max_y)

    def bbox_center(self):
        ""
        Returns 2D point that is in the middle of the bounding box
        defined by polygon points

        :return: tuple that contains coordinates of 2D point that is in
                 the middle of the bounding box
        ""
        left, bottom, right, top = self.bbox_points
        width = abs(right - left)
        height = abs(top - bottom)
        center_x = int(left + width / 2)
        center_y = int(top - height / 2)

        return center_x, center_y
"""

"""
class Table(object):
    ""
    Class used for loading data from the OCR output.

    This class is later used by data loader to create graph
    representation of the OCR output that can be used as
    input for the trained model.

    :type file_name_path:      String
    :parameter file_name_path: Path to file containing XML representation
                               of the OCR output.
    ""

    def __init__(self, file_name_path):
        ""
        The constructor of Table class.

        :type file_name_path:      String
        :parameter file_name_path: Path to file containing XML representation
                                   of the OCR output.
        ""
        self.file_name_path = file_name_path

        self.parse_ocr_output()

    def parse_ocr_output(self):
        ""
        Function that parses the XML representation of the OCR.

        :return: None
        ""
        xml_root = ET.parse(self.file_name_path)
        print(xml_root)

"""
"""
class TableDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data_1.pt', 'data_2.pt', ...]

    def download(self):
        # Download to `self.raw_dir`.
        path = download_url(url, self.raw_dir)
        ...

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            data = Data(...)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data
"""
