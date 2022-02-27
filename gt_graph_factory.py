import os
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

            x_coords = [x for x, _ in text_line.polygon_points]
            y_coords = [y for _, y in text_line.polygon_points]
            max_x, min_x = max(x_coords), min(x_coords)
            max_y, min_y = max(y_coords), min(y_coords)
            text_line.bbox_points = (min_x, min_y, max_x, max_y)
            idx.insert(text_line.rtree_id, text_line.bbox_points)

        # Create graph representation
        for text_line in self.text_lines:
            k_neighbors = list(idx.nearest(text_line.bbox_points, k_param))
            self.edges = self.edges.union({(text_line.rtree_id, neighbor) for neighbor in k_neighbors})
            self.edges = self.edges.union({(neighbor, text_line.rtree_id) for neighbor in k_neighbors})

    def visualize_graph(self, visualize_dir="./dataset_preparation/visualization"):
        """
        Visualize the created graph. This function MUST be called after one of the
        functions for graph creation create_*

        :type visualize_dir:  String
        :param visualize_dir: Directory that should store the visualizations
        """

        img = cv2.imread(self.dataset_image_path)

        for text_line in self.text_lines:
            center_x, center_y = text_line.bbox_center()
            cv2.circle(img, (center_x, center_y),  radius=10, color=(0, 0, 255), thickness=-1)

            neighbors = [edge_dst for edge_src, edge_dst in self.edges if edge_src == text_line.rtree_id]

            for neighbor in neighbors:
                src_pt = self.rtree_id_to_text_line[text_line.rtree_id].bbox_center()
                dst_pt = self.rtree_id_to_text_line[neighbor].bbox_center()
                cv2.line(img, src_pt, dst_pt, color=(0, 0, 0), thickness=3)

        image_name = self.dataset_image_path.split("/")[-1]
        cv2.imwrite(os.path.join(visualize_dir, "GRAPH_" + image_name), img)


class OCRTextLine(object):
    """
    Class representing a given text line.

    The class takes as an input XML element representing given text line.
    The XML element is parsed by the class and attributes of the element
    are stored inside the object.

    :type text_line_xml:  XMLElement
    :param text_line_xml: XML element representing coordinates of a given text line
    """

    def __init__(self, text_line_xml):
        """
        A constructor for OCRNode class

        :param text_line_xml: XML element representing a given text line

        """
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
        center_y = int(top + height / 2)

        return center_x, center_y


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
