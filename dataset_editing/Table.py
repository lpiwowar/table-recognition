import copy

import cv2
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom

from Cell import Cell


class Table(object):
    def __init__(self, gt_path=None, img_path=None, gt_type="XML"):
        self.img_path = img_path
        self.gt_path = gt_path
        self.cells = []
        self.table_region = []

        if gt_type == "XML":
            self.xml_parse_gt()

        return

    def __str__(self):
        return f"<Table img_path:{self.img_path} gt_path:{self.gt_path} " \
               f"table_region:{self.table_region}>"

    def fix_table(self):
        header_max_value = [int(cell.end_row) for cell in self.cells if cell.type == "header"]
        data_min_value = [int(cell.start_row) for cell in self.cells if cell.type == "data"]

        if not header_max_value or not data_min_value:
            return

        header_max_value = max(header_max_value)
        data_min_value = min(data_min_value)
        if not header_max_value <= data_min_value:
            data_cells = [cell for cell in self.cells if cell.type == "data"]
            diff = header_max_value - data_min_value
            for data_cell in data_cells:
                data_cell.start_row = str(diff + int(data_cell.start_row))
                data_cell.end_row = str(diff + int(data_cell.end_row))

    def table_region_to_string(self):
        coords = [[str(tuple[0]), str(tuple[1])] for tuple in self.table_region]
        coords = [",".join(tuple) for tuple in coords]
        coords = " ".join(coords)
        return coords

    def get_xml_eltree(self):
        document = ET.Element("document")
        table = ET.Element("table")
        coords = ET.Element("Coords", {"points": self.table_region_to_string()})
        table.append(coords)
        for cell in self.cells:
            table.append(cell.get_xml_eltree())
        document.append(table)
        return document

    def get_xml_string(self):
        xml_string = ET.tostring(self.get_xml_eltree(), encoding="utf8", method="xml")
        xml_string = minidom.parseString(xml_string).toprettyxml()
        return xml_string

    def xml_parse_gt(self):
        tree = ET.parse(self.gt_path)
        root = tree.getroot()

        # Parse table region
        # xml_table_coords = root[0].findall("Coords")[0]
        # self.table_region = Table.xml_parse_coord(xml_table_coords.attrib["points"])

        # Parse individual cells
        xml_cells = root[0].findall("cell")
        for xml_cell in xml_cells:
            start_row = xml_cell.attrib["start-row"]
            end_row = xml_cell.attrib["end-row"]
            start_col = xml_cell.attrib["start-col"]
            end_col = xml_cell.attrib["end-col"]
            cell_coords = Table.xml_parse_coord(xml_cell.find("Coords").attrib["points"])
            cell = Cell(start_row, end_row, start_col, end_col, cell_coords, None)
            cell.type = xml_cell.attrib["type"]
            self.cells.append(cell)

        return

    @staticmethod
    def xml_parse_coord(coord_string):
        parsed_coords = coord_string.split(" ")
        parsed_coords = [tuple_coord.split(",") for tuple_coord in parsed_coords]
        return parsed_coords

    def annotate_table(self):
        # Load image
        image = cv2.imread(self.img_path)

        # Rescale image
        scale_factor = 0.20
        width = int(image.shape[1] * scale_factor)
        height = int(image.shape[0] * scale_factor)
        image = cv2.resize(image, (width,height), interpolation=cv2.INTER_AREA)

        end_annotation = False
        for cell in self.cells:
            # Rescale coordinates to rescaled image
            coords = np.array(cell.coords, np.int32)
            coords = np.array(np.ceil(coords * scale_factor), np.int32)

            pts = coords.reshape((-1, 1, 2))
            isClosed = True
            color = (0, 0, 0)
            thickness = 2

            image = cv2.polylines(image, [coords], isClosed, color, thickness)
            while (1):
                cv2.imshow('image', image)
                k = cv2.waitKey(33)
                if k == 27:
                    break
                elif k == ord('w'):
                    end_annotation = True
                    break
                elif k == ord('d'):
                    cell.type = "data"
                    color = (255, 0, 0)
                    image = cv2.polylines(image, [pts], isClosed, color, thickness)
                    break
                elif k == ord('h'):
                    cell.type = "header"
                    color = (0, 255, 0)
                    image = cv2.polylines(image, [pts], isClosed, color, thickness)
                    break
                elif k == ord('c'):
                    cell.type = "data_empty"
                    color = (0, 255, 255)
                    image = cv2.polylines(image, [pts], isClosed, color, thickness)
                    break
                elif k == ord('x'):
                    cell.type = "data_mark"
                    color = (0, 0, 255)
                    image = cv2.polylines(image, [pts], isClosed, color, thickness)
                    break
                elif k == ord('b'):
                    return True

            if end_annotation:
                break

        cv2.destroyAllWindows()
        return False

    def display_table(self):
        # Load image
        image = cv2.imread(self.img_path)

        # Rescale image
        scale_percent = 0.20
        width = int(image.shape[1] * scale_percent)
        height = int(image.shape[0] * scale_percent)
        image = cv2.resize(image, (width,height), interpolation=cv2.INTER_AREA)

        image_save = copy.deepcopy(image)

        for cell in self.cells:
            coords = np.array(cell.coords, np.int32)
            coords = np.array(np.ceil(coords * scale_percent), np.int32)

            coords = coords.reshape((-1, 1, 2))
            isClosed = True
            if not cell.type:
                color = (0, 0, 0)
            elif cell.type == "data":
                color = (255, 0, 0)
            elif cell.type == "header":
                color = (0, 255, 0)
            elif cell.type == "empty":
                color = (0, 0, 255)
            else:
                color = (0, 0, 0)

            thickness = 2
            image = cv2.polylines(image, [coords],
                                  isClosed, color, thickness)

        while (1):
            cv2.imshow('image', image)
            k = cv2.waitKey(33)
            if k == ord('e'):
                end = True
                break

        cv2.destroyAllWindows()
        return
