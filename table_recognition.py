import argparse
import datetime
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils.metric import accuracy
from tqdm import tqdm
import wandb

from config_parser import Config
from dataset import TableDataset
from graph import Graph
from model import SimpleModel
from utils import visualize_graph
from utils import visualize_model_output
from utils import visualize_output_image
from utils import visualize_input_image


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


def train(conf):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel().to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.NLLLoss()

    table_dataset = TableDataset(conf)

    train_size = int(0.8 * len(table_dataset))
    test_size = len(table_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(table_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)

    WANDB = True
    if WANDB:
        wandb.init(project="table-recognition",
                   name=datetime.datetime.now().strftime("%Y-%m-%d-%H:%M"),
                   entity="lpiwowar")
        wandb.watch(model)

    average_loss_edges, average_loss_nodes = [], []
    counter = 0
    best_accuracy_edges = 0

    LOAD_MODEL = False
    if LOAD_MODEL:
        model.load_state_dict(torch.load(conf.model_path))

    for epoch in tqdm(range(100)):
        for data in train_loader:
            optimizer.zero_grad()
            out_nodes, out_edges = model(data)

            y, edge_output_attr = torch.argmax(data.y, dim=1), torch.argmax(data.edge_output_attr, dim=1)
            loss_nodes = criterion(out_nodes, y)
            loss_edges = criterion(out_edges, edge_output_attr)

            # loss_nodes.backward(retain_graph=True)
            loss_edges.backward()

            counter += 1
            average_loss_edges.append(loss_edges)
            average_loss_nodes.append(loss_nodes)
            if counter > 50:
                # Log loss
                counter = 0
                avg_edges = sum(average_loss_edges) / len(average_loss_edges)
                avg_nodes = sum(average_loss_nodes) / len(average_loss_nodes)
                wandb.log({"loss_edges": avg_edges, "loss_nodes": avg_nodes})
                average_loss_edges, average_loss_nodes = [], []

            optimizer.step()

        # Log accuracy
        for data in test_loader:
            out_nodes, out_edges = model(data)
            out_nodes, out_edges = torch.argmax(torch.exp(out_nodes), dim=1), torch.argmax(torch.exp(out_edges), dim=1)

            exp_out_nodes = torch.argmax(data.y, dim=1)
            exp_out_edges = torch.argmax(data.edge_output_attr, dim=1)

            accuracy_nodes = accuracy(exp_out_nodes, out_nodes)
            accuracy_edges = accuracy(exp_out_edges, out_edges)
            wandb.log({"accuracy_nodes": accuracy_nodes, "accuracy_edges": accuracy_edges})

            if accuracy_edges > best_accuracy_edges:
                torch.save(model.state_dict(), conf.model_path)
                best_accuracy_edges = accuracy_edges

    print(f"Best accuracy: {best_accuracy_edges}")

    model.load_state_dict(torch.load(conf.model_path))

    for data in test_loader:
        out_nodes, out_edges = model(data)
        out_nodes, out_edges = torch.argmax(torch.exp(out_nodes), dim=1), torch.argmax(torch.exp(out_edges), dim=1)
        visualize_output_image(data, out_nodes, out_edges, conf.visualize_path)
        visualize_input_image(data, "/home/lpiwowar-personal/PycharmProjects/master-thesis/input_images_test")


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
    elif args.train:
        train(config)

