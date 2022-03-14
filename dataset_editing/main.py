import argparse
import os

from Table import Table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Program that helps to add classification to table cells.')
    parser.add_argument('--gt_dir', type=str)
    parser.add_argument('--img_dir', type=str)
    args = parser.parse_args()
    gt_dir_path = args.gt_dir
    img_dir_path = args.img_dir
    gt_filenames = os.listdir(gt_dir_path)
    img_filenames = os.listdir(img_dir_path)
    gt_filenames.sort()
    img_filenames.sort()

    for gt_filename, img_filename in zip(gt_filenames, img_filenames):
        print(f"{gt_filename} + {img_filename}")
        table = Table(gt_dir_path + gt_filename, img_dir_path + img_filename)
        while table.annotate_table():
            pass

        with open("./output/" + gt_filename, "w") as f:
            f.write(table.get_xml_string())
