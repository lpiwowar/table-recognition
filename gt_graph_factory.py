import os
from rtree import index
from xml.etree import ElementTree

from utils import coords_string_to_tuple_list

# import torch
# from torch_geometric.data import Dataset


class GTGraphCreator(object):
    """
    Class that creates ground truth graph representation.

    This class uses output from the OCR and XML CTDAR ground truth
    data to create the graph representation of the expected output
    of the GNN.

    :type config:  Config
    :param config: Instance of object Config containing configuration
                   information
    """

    def __init__(self, config):
        self.config = config
        self.text_lines = []

    def create_k_nearest_neighbors_graphs(self):
        """
        Create graph representation of XML CTDAR ground truth data.
        The created graph representation can be used to train the GNN.
        """

        ns = {"xmlns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15",
              "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
              "xsi:schemaLocation": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd"}

        xpath_coords = "./xmlns:Page/xmlns:TextRegion/xmlns:TextLine"

        # Parse XML OCR output
        for ocr_file in os.listdir(self.config.dataset_gt_path):
            ocr_file_path = os.path.join(self.config.ocr_output_path, ocr_file)
            coords_xml = ElementTree.parse(ocr_file_path).findall(xpath_coords, ns)
            self.text_lines += [OCRTextLine(coord_xml) for coord_xml in coords_xml]

        # Create Rtree index
        idx = index.Index()
        rtree_id = 0
        for text_line in self.text_lines:
            text_line.rtree_id = rtree_id
            rtree_id += 1

            x_coords = [x for x, _ in text_line.polygon_points]
            y_coords = [y for _, y in text_line.polygon_points]
            max_x, min_x = max(x_coords), min(x_coords)
            max_y, min_y = max(y_coords), min(y_coords)
            idx.insert(text_line.rtree_id, (min_x, min_y, max_x, max_y))


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
        self.text_line_xml = text_line_xml   # XML representation of TextLine
        self.uuid = None                     # UUID that represents the text line in XML
        self.rtree_id = None                 # ID that represents the polygon in rtree db
        self.polygon_points = []             # List of 2D coords that define the text line polygon

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
