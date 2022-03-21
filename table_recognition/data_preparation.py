import os

from tqdm import tqdm
from table_recognition.graph import Graph


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

    for ocr_file in tqdm(ocr_files, disable=conf.tqdm_disable):
        ocr_file_prefix = ocr_file.split(".")[0]
        if ocr_file_prefix + ".xml" not in gt_files:
            raise Exception(f"ERROR: {ocr_file_prefix + '.xml'} is missing in the dataset GT dir.")
        if ocr_file_prefix + ".jpg" not in img_files:
            raise Exception(f"ERROR: {ocr_file_prefix + '.jpg'} is missing in the dataset IMG dir.")

        ocr_file_path = os.path.join(conf.ocr_output_path, ocr_file)
        dataset_gt_path = os.path.join(conf.dataset_gt_path, ocr_file_prefix + '.xml')
        dataset_img_path = os.path.join(conf.dataset_img_path, ocr_file_prefix + '.jpg')

        graph = Graph(conf, ocr_file_path, dataset_gt_path, dataset_img_path,
                      input_graph_colorer=conf.input_graph_colorer)

        graph.initialize()
        graph.color_output()
        graph.color_input()
        graph.visualize()
        graph.dump()