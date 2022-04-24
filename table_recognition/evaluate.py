import copy
import numpy as np
import os
import xml.etree.ElementTree as ET

from rtree import index
from shapely.geometry import Polygon

from table_recognition.graph.utils import coords_string_to_tuple_list
from table_recognition.graph.colorers.utils import get_multiple_values_from_dict


def evaluate(config):
    infer_ctdar_xml_files = os.listdir(config.infer_xml_output)
    infer_ctdar_xml_files.sort()

    gt_ctdar_xml_files = os.listdir(config.gt_xml)
    gt_ctdar_xml_files.sort()

    for infer_ctdar_xml_file in infer_ctdar_xml_files:
        if infer_ctdar_xml_file not in gt_ctdar_xml_files:
            config.logger.info(f"Skipping: {infer_ctdar_xml_file} due to missing ground truth file in {config.gt_xml}")
            continue

        infer_ctdar_xml_file_path = os.path.join(config.infer_xml_output, infer_ctdar_xml_file)
        gt_xml_file_path = os.path.join(config.gt_xml, infer_ctdar_xml_file)
        Evaluate(infer_ctdar_xml_file_path, config.infer_xml_output_type,
                 gt_xml_file_path, config.gt_xml_type)


class Evaluate(object):
    CELL_DETECTION_IOU_THRESHOLD = 0.6

    def __init__(self, infer_xml_file, infer_xml_file_type, gt_xml_file, gt_xml_file_type):
        self.infer_xml_file = infer_xml_file
        self.infer_xml_file_type = infer_xml_file_type
        self.gt_xml_file = gt_xml_file
        self.gt_xml_file_type = gt_xml_file_type

        self.infer_cells = self.parse_xml_cells(self.infer_xml_file, self.infer_xml_file_type)
        self.id_to_infer_cells = {cell.id: cell for cell in self.infer_cells}
        self.infer_cells_grid = self.create_cells_grid(self.infer_cells)

        self.gt_cells = self.parse_xml_cells(self.gt_xml_file, self.gt_xml_file_type)
        self.id_to_gt_cells = {cell.id: cell for cell in self.gt_cells}
        self.gt_cells_grid = self.create_cells_grid(self.gt_cells)

        self.gt_cells_rtree_index = index.Index()
        self.populate_rtree_index(self.gt_cells_rtree_index, self.gt_cells)

        self.mark_incorrectly_detected_cells()
        self.correctly_detected_cells_percentage = self.calc_correctly_detected_cells()
        self.precisely_detected_cells_percentage = self.calc_precisely_detected_cells()
        print(f"infer_xml_file: {infer_xml_file} correct_cells: {self.correctly_detected_cells_percentage} "
              f"precisely_cells: {self.precisely_detected_cells_percentage} "
              f"horizontal_neighbours: {self.calc_horizontal_neighbours()} "
              f"vertical_neighbours: {self.calc_vertical_neighbours()}")

    def get_horizontal_neighbours(self, cell):
        cell_row = self.infer_cells_grid[cell.start_row]

        if cell.start_col + 1 < len(cell_row):
            row_right_to_cell = self.infer_cells_grid[cell.start_row][cell.start_col+1:]
        else:
            row_right_to_cell = None

        if cell.start_col < len(cell_row):
            row_left_to_cell = self.infer_cells_grid[cell.start_row][:cell.start_col]
        else:
            row_left_to_cell = None

        first_right_cell = [cell for cell in row_right_to_cell if cell is not None]
        first_right_cell = first_right_cell[0] if first_right_cell else None

        first_left_cell = [cell for cell in row_left_to_cell if cell is not None]
        first_left_cell = first_left_cell[-1] if first_left_cell else None

        return [first_left_cell, cell, first_right_cell]

    def get_cells_gt_coords(self, cells):
        gt_coords = []
        for cell in cells:
            if not cell:
                continue

            intersection_ids = list(self.gt_cells_rtree_index.intersection(cell.bbox["rtree"]))
            gt_cells = get_multiple_values_from_dict(self.id_to_gt_cells, intersection_ids)

            new_gt_coords = (None, None)
            for gt_cell in gt_cells:
                intersection_value = self.calc_intersection(cell, gt_cell)
                if intersection_value > Evaluate.CELL_DETECTION_IOU_THRESHOLD:
                    new_gt_coords = (gt_cell.start_row, gt_cell.start_col)

            gt_coords += [new_gt_coords]

        return gt_coords

    def calc_horizontal_neighbours(self):
        correct_counter = 0
        for cell in self.infer_cells:
            cell_horizontal_neighbours = self.get_horizontal_neighbours(cell)
            cell_gt_coords = self.get_cells_gt_coords(cell_horizontal_neighbours)
            gt_coords_row_idxs = [start_row for (start_row, start_col) in cell_gt_coords]
            in_same_row = all([value == gt_coords_row_idxs[0] for value in gt_coords_row_idxs])
            if in_same_row:
               correct_counter += 1

        return correct_counter / len(self.infer_cells)

    def get_vertical_neighbours(self, cell):
        infer_cells_grid = copy.copy(self.infer_cells_grid)
        infer_cells_grid = np.array(infer_cells_grid).T

        cell_col = infer_cells_grid[cell.start_col]

        if cell.start_row + 1 < len(cell_col):
            col_bot_to_cell = infer_cells_grid[cell.start_col][cell.start_row+1:]
        else:
            col_bot_to_cell = None

        if cell.start_row < len(cell_col):
            col_top_to_cell = infer_cells_grid[cell.start_col][:cell.start_row]
        else:
            col_top_to_cell = None

        first_bot_cell = [cell for cell in col_bot_to_cell if cell is not None]
        first_bot_cell = first_bot_cell[0] if first_bot_cell else None

        first_top_cell = [cell for cell in col_top_to_cell if cell is not None]
        first_top_cell = first_top_cell[-1] if first_top_cell else None

        return [first_top_cell, cell, first_bot_cell]

    def calc_vertical_neighbours(self):
        correct_counter = 0
        for cell in self.infer_cells:
            cell_vertical_neighbours = self.get_vertical_neighbours(cell)
            cell_gt_coords = self.get_cells_gt_coords(cell_vertical_neighbours)
            gt_coords_col_idxs = [start_col for (start_row, start_col) in cell_gt_coords]

            in_same_row = all([value == gt_coords_col_idxs[0] for value in gt_coords_col_idxs])
            if in_same_row:
                correct_counter += 1

        return correct_counter / len(self.infer_cells)

    def create_cells_grid(self, cells):
        max_cols = max([cell.end_col for cell in cells])
        max_rows = max([cell.end_row for cell in cells])
        grid = [[None for _ in range(max_cols+1)] for _ in range(max_rows+1)]

        for cell in cells:
            grid[cell.start_row][cell.start_col] = cell

        return grid

    def calc_correctly_detected_cells(self):
        correctly_detected = [cell for cell in self.infer_cells if cell.correctly_detected]
        return len(correctly_detected) / len(self.gt_cells)

    def calc_precisely_detected_cells(self):
        correctly_detected = [cell for cell in self.infer_cells if cell.correctly_detected]
        return len(correctly_detected) / len(self.infer_cells)

    def mark_incorrectly_detected_cells(self):
        for cell in self.infer_cells:
            intersection_ids = list(self.gt_cells_rtree_index.intersection(cell.bbox["rtree"]))
            gt_cells = get_multiple_values_from_dict(self.id_to_gt_cells, intersection_ids)

            for gt_cell in gt_cells:
                intersection_value = self.calc_intersection(cell, gt_cell)
                if intersection_value > Evaluate.CELL_DETECTION_IOU_THRESHOLD:
                    cell.correctly_detected = True

    def calc_intersection(self, cell1, cell2):
        cell1_polygon = Polygon(cell1.bbox["polygon"])
        cell2_polygon = Polygon(cell2.bbox["polygon"])
        intersection = cell1_polygon.intersection(cell2_polygon).area
        return intersection / cell1_polygon.area

    def populate_rtree_index(self, rtree_index, cells):
        for cell in cells:
            rtree_index.insert(cell.id, cell.bbox["rtree"])

    def parse_xml_cells(self, xml_file, xml_file_type):
        output_cells = []

        if xml_file_type == "ICDAR":
            tree = ET.parse(xml_file)
            cells_xml = tree.findall("./table/cell")
            for cell_xml in cells_xml:
                coords_xml = cell_xml.find("./Coords")
                output_cells += [Cell(start_row=int(cell_xml.attrib["start-row"]),
                                      end_row=int(cell_xml.attrib["end-row"]),
                                      start_col=int(cell_xml.attrib["start-col"]),
                                      end_col=int(cell_xml.attrib["end-col"]),
                                      points=coords_string_to_tuple_list(coords_xml.attrib["points"]))]
        elif xml_file_type == "ABP":
            tree = ET.parse(xml_file)
            table_cells_xml = tree.findall("./Page/TableCell")
            for table_cell_xml in table_cells_xml:
                coords_xml = table_cell_xml.find("./Coords")
                output_cells += [Cell(start_row=int(table_cell_xml.attrib["row"]),
                                      end_row=int(table_cell_xml.attrib["row"]) + int(table_cell_xml.attrib["rowSpan"]),
                                      start_col=int(table_cell_xml.attrib["col"]),
                                      end_col=int(table_cell_xml.attrib["col"]) + int(table_cell_xml.attrib["colSpan"]),
                                      points=coords_string_to_tuple_list(coords_xml.attrib["points"]))]
        else:
            raise Exception(f"ERROR - {xml_file_type} file type not supported!")

        return output_cells


class Cell(object):
    CELL_ID_COUNTER = 0

    def __init__(self, start_row, end_row, start_col, end_col, points):
        self.start_row = start_row
        self.end_row = end_row
        self.start_col = start_col
        self.end_col = end_col
        self.bbox = self.calculate_bbox(points)

        self.id = Cell.CELL_ID_COUNTER
        Cell.CELL_ID_COUNTER += 1

        self.correctly_detected = False

    def __repr__(self):
        return f"<Cell start_row={self.start_row} " \
               f"end_row={self.end_row} " \
               f"start_col={self.start_col} " \
               f"end_col={self.end_col} " \
               f'bbox={self.bbox["corners"]}>'

    def calculate_bbox(self, points):
        x_coords = [x for x, _ in points]
        y_coords = [y for _, y in points]
        max_x, min_x = max(x_coords), min(x_coords)
        max_y, min_y = max(y_coords), min(y_coords)

        bbox_coord_types = {
            "rtree": (min_x, min_y, max_x, max_y),
            "polygon": [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)],
            "corners": [(min_x, min_y), (max_x, max_y)],
            "center": [int(min_x + ((max_x - min_x) / 2)), int(min_y + ((max_y - min_y) / 2))]
        }

        return bbox_coord_types
