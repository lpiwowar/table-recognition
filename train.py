import datetime

import torch
import wandb
from torch_geometric.loader import DataLoader
from torch_geometric.utils.metric import accuracy
from tqdm import tqdm

from dataset import TableDataset
from model import SimpleModel
from utils import visualize_output_image
from utils import visualize_input_image


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

