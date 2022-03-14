import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

import cv2
import numpy as np
from tqdm import tqdm

from utils import coords_string_to_tuple_list, tuple_list_to_coords_string

GT_DIR = "../dataset/cTDaR/ground_truth_renamed"
IMAGE_DIR = "../dataset/cTDaR/image_jpg"

GT_DIR_OUTPUT = "../dataset/cTDaR/ground_truth_cropped"
IMAGE_DIR_OUTPUT = "../dataset/cTDaR/image_cropped"

gt_files = os.listdir(GT_DIR)
img_files = os.listdir(IMAGE_DIR)

"""
def filter_out_polygon(img_name, polygon_pts):
    ""
    Source: https://www.semicolonworld.com/question/56606/numpy-opencv-2-how-do-i-crop-non-rectangular-region
    ""
    # original image
    # -1 loads as-is so if it will be 3 or 4 channel as the original
    image = cv2.imread(img_name, -1)
    # mask defaulting to black for 3-channel and transparent for 4-channel
    # (of course replace corners with yours)
    mask = np.zeros(image.shape, dtype=np.uint8)
    roi_corners = np.array(polygon_pts, dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,) * channel_count
    # print(f"roi_corners: {roi_corners} ignore_mask_color: {ignore_mask_color}")
    cv2.fillPoly(mask, pts = [roi_corners], color = ignore_mask_color)
    # cv2.fillPoly(image, pts = [roi_corners], color = ignore_mask_color)
    # from Masterfool: use cv2.fillConvexPoly if you know it's convex

    # apply the mask
    masked_image = cv2.bitwise_and(image, mask)

    # save the result
    return image
"""


def filter_out_polygon(img_name, polygon_pts):
    polygon_pts = [(abs(x), abs(y)) for x, y in polygon_pts]
    img = cv2.imread(img_name)
    pts = np.array(polygon_pts)

    ## (1) Crop the bounding rect
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    croped = img[y:y + h, x:x + w].copy()

    ## (2) make mask
    pts = pts - pts.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    ## (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)

    ## (4) add the white background
    #bg = np.ones_like(croped, np.uint8) * 255
    #cv2.bitwise_not(bg, bg, mask=mask)
    #dst2 = bg + dst

    # cv2.imwrite("croped.png", croped)
    # cv2.imwrite("mask.png", mask)
    # cv2.imwrite("dst.png", dst)
    # cv2.imwrite("dst2.png", dst2)

    return dst


def get_bbox(polygon_coords):
    x_coords = [x for x, _ in polygon_coords]
    y_coords = [y for _, y in polygon_coords]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    return [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]

"""
# Image cropping
for gt_file in tqdm(gt_files):
    tree = ET.parse(os.path.join(GT_DIR, gt_file))
    tables = tree.findall("./table")
    table_idx = 0
    for table in tables:
        table_idx += 1
        table_polygon_coords = table.find("Coords").attrib["points"]
        table_polygon_coords = coords_string_to_tuple_list(table_polygon_coords)
        table_bbox = get_bbox(table_polygon_coords)

        gt_file_prefix = gt_file.split("/")[-1].split(".")[-2]
        img_name = gt_file_prefix + ".jpg"
        img_name_path = os.path.join(IMAGE_DIR, img_name)

        if img_name in img_files:
            image = filter_out_polygon(img_name_path, table_polygon_coords)
        else:
            print(f"{img_name} does not exist!")
            continue

        cv2.imwrite(os.path.join(IMAGE_DIR_OUTPUT, gt_file_prefix) + f"_table_{table_idx}.jpg", image)
"""

for gt_file in tqdm(gt_files):
    tree = ET.parse(os.path.join(GT_DIR, gt_file))
    tables = tree.findall("./table")
    table_idx = 0
    for table in tables:
        table_idx += 1
        table_coords = table.find("./Coords").attrib["points"]
        table_coords = coords_string_to_tuple_list(table_coords)
        [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)] = get_bbox(table_coords)
        
        if gt_file == "cTDaR_t00002.xml":
            print(f"gt_file: {gt_file} min_x: {min_x}  min_y: {min_y}")

        cells = table.findall("./cell")

        new_document_xml = ET.Element("document")
        new_table_xml = ET.Element("table")

        for cell in cells:
            coords_xml = cell.find("./Coords")
            cell_polygon_coords = coords_string_to_tuple_list(coords_xml.attrib["points"])
            cell_polygon_coords = [(x - min_x, y - min_y) for x, y in cell_polygon_coords]
            cell_polygon_coords = [(max(0, x), max(0, y)) for x, y in cell_polygon_coords]
            cell_polygon_coords = tuple_list_to_coords_string(cell_polygon_coords)

            new_coords_xml = ET.Element("Coords")
            new_coords_xml.set("points", cell_polygon_coords)

            cell.remove(coords_xml)
            cell.append(new_coords_xml)
            new_table_xml.append(cell)

        new_document_xml.append(new_table_xml)

        xml_string = ET.tostring(new_document_xml, encoding="utf8", method="xml")
        xml_string = minidom.parseString(xml_string).toprettyxml()

        gt_file_prefix = gt_file.split("/")[-1].split(".")[-2]

        with open(os.path.join(GT_DIR_OUTPUT, gt_file_prefix) + f"_table_{table_idx}.xml", "w") as f:
            f.write(xml_string)
