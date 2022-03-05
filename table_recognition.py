import argparse
import os

from tqdm import tqdm

from config_parser import Config
from graph import Graph


def data_preparation(conf):
    """
    Prepare graph representation that can be used to train the GNN.

    :type conf:  Config
    :param conf: Instance of Config class that contains configuration
                 information
    """
    ocr_files = [file for file in os.listdir(conf.ocr_output_path)]
    gt_files = [file for file in os.listdir(conf.dataset_gt_path)]
    img_files = [file for file in os.listdir(conf.dataset_img_path)]

    for ocr_file in tqdm(ocr_files):
        ocr_file_prefix = ocr_file.split(".")[0]
        if ocr_file_prefix + ".xml" not in gt_files:
            raise Exception(f"ERROR: {ocr_file_prefix + '.xml'} is missing in the dataset GT dir.")
        if ocr_file_prefix + ".jpg" not in img_files:
            raise Exception(f"ERROR: {ocr_file_prefix + '.jpg'} is missing in the dataset IMG dir.")

        ocr_file_path = os.path.join(config.ocr_output_path, ocr_file)
        dataset_gt_path = os.path.join(config.dataset_gt_path, ocr_file_prefix + '.xml')
        dataset_img_path = os.path.join(config.dataset_img_path, ocr_file_prefix + '.jpg')

        graph = Graph(conf, ocr_file_path, dataset_gt_path, dataset_img_path)
        graph.initialize()
        graph.color_output()
        graph.color_input()
        graph.visualize()
        graph.dump()


def check_arguments(arg):
    """
    A function that checks whether valid combination of
    arguments is used.

    :type arg:  ArgumentParser
    :param arg: Parsed arguments that should be checked.
    :return:     True when valid combination of arguments is used
                 False otherwise
    """
    return not (not arg.train ^ arg.infer) ^ arg.data_preparation


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Table recognition")
    parser.add_argument("--train",
                        help="Train model for table recognition (--config-file required)",
                        action="store_true")
    parser.add_argument("--infer",
                        help="Use trained model for table recognition (--config-file required)",
                        action="store_true")
    parser.add_argument("--data-preparation",
                        help="Prepare dataset for training (--config-file required)",
                        action="store_true")
    parser.add_argument("--config-file",
                        help="Path to configuration file",
                        default="./table_recognition_config.ini")
    args = parser.parse_args()

    if not check_arguments(args):
        raise Exception("ERROR: Either --train, --infer or --data-preparation must be specified.")

    config = Config(args.config_file)

    if args.data_preparation:
        data_preparation(config)
    # for file in os.listdir(config.train_input_data_dir):
    #   print(file)
