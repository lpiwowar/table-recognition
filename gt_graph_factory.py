import os
from operator import itemgetter
from xml.etree import ElementTree

import cv2
from rtree import index

from utils import coords_string_to_tuple_list


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

        # Create Rtree index
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
            self.edges = self.edges.union({(text_line.rtree_id, neighbor) for neighbor in k_neighbors})
            self.edges = self.edges.union({(neighbor, text_line.rtree_id) for neighbor in k_neighbors})

            # Define type of node in graph
            cells_intersection = list(idx_cells.intersection(text_line.bbox_points))
            if cells_intersection:
                dataset_gt_nodes = self.get_dataset_gt_nodes_by_rtree_idxs(cells_intersection)
                cells_intersection_types = [cell.type for cell in dataset_gt_nodes]
                cell_type = max(cells_intersection_types, key=cells_intersection_types.count)
                text_line.type = cell_type

    def get_dataset_gt_nodes_by_rtree_idxs(self, rtree_idxs):
        """
        A function that takes as an input a list of rtree_idxs and
        returns a list of corresponding dataset_gt_objects

        :type rtree_idxs:  List of integers
        :param rtree_idxs: List of rtree indeces
        :return:           List of DatasetGTNode objects
        """
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
                None: (0, 0, 0)
            },
            "edge": {

            }
        }

        img = cv2.imread(self.dataset_image_path)

        for text_line in self.text_lines:
            # Draw polygon that defines region for a given text line
            left, bottom, right, top = text_line.bbox_points
            cv2.rectangle(img, (left, bottom), (right, top), color=(0, 0, 255), thickness=2)

            # Draw edges for a given node
            neighbors = [edge_dst for edge_src, edge_dst in self.edges if edge_src == text_line.rtree_id]
            for neighbor in neighbors:
                src_pt = self.rtree_id_to_text_line[text_line.rtree_id].bbox_center()
                dst_pt = self.rtree_id_to_text_line[neighbor].bbox_center()
                cv2.line(img, src_pt, dst_pt, color=(0, 0, 0), thickness=3)

            # Draw graphs node (circle in the middle of the text line)
            center_x, center_y = text_line.bbox_center()
            cv2.circle(img, (center_x, center_y), radius=10, color=colors["node"][text_line.type], thickness=-1)

        image_name = self.dataset_image_path.split("/")[-1]
        cv2.imwrite(os.path.join(visualize_dir, "GRAPH_" + image_name), img)


class Edge(object):
    def __init__(self):
        self.node_left = None
        self.node_right = None
        self.type = None


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

        self.text_line_xml = text_line_xml
        self.polygon_points = None
        self.bbox_points = None
        self.rtree_id = None

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
