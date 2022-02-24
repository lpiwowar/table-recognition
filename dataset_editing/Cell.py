import xml.etree.ElementTree as ET


class Cell(object):
    def __init__(self, start_row, end_row, start_col, end_col, coords, type):
        self.start_row = start_row
        self.end_row = end_row
        self.start_col = start_col
        self.end_col = end_col
        self.coords = coords
        self.type = type

    def __str__(self):
        return f"<Cell start_row:{self.start_row} end_row:{self.end_row} " \
               f"start_col:{self.start_col} end_col:{self.end_col} " \
               f"coords:{self.coords} type:{self.type}>"

    def get_xml_eltree(self):
        kwargs = {
            "start-row": self.start_row,
            "end-row": self.end_row,
            "start-col": self.start_col,
            "end-col": self.end_col,
        }
        cell = ET.Element("cell", kwargs)
        coords = ET.Element("Coords", {'points': self.coords_to_string()})
        cell.append(coords)
        return cell

    def coords_to_string(self):
        coords = [[str(tuple[0]), str(tuple[1])] for tuple in self.coords]
        coords = [",".join(tuple) for tuple in coords]
        coords = " ".join(coords)
        return coords



