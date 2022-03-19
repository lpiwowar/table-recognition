import datetime
import logging

import torch
import wandb
from torch_geometric.loader import DataLoader
from torch_geometric.utils.metric import accuracy
from tqdm import tqdm

from table_recognition.dataset import TableDataset
from table_recognition.models import SimpleModel
from table_recognition.graph.utils import visualize_output_image
from table_recognition.graph.utils import visualize_input_image


def make(hyperparams, conf):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel().to(device)

    if conf.preload_model:
        model.load_state_dict(torch.load(conf.model_path))

    table_dataset = TableDataset(conf)

    train_size = int(0.8 * len(table_dataset))
    test_size = len(table_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(table_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=hyperparams.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.NLLLoss()

    return model, train_loader, test_loader, criterion, optimizer


def train(model, train_loader, test_loader, criterion, optimizer, hyperparams, conf):
    wandb.watch(model, criterion, log="all", log_freq=10)

    batch_counter = 0
    example_counter = 0
    best_accuracy = 0
    for epoch in tqdm(range(hyperparams.epochs), disable=conf.tqdm_disable):
        conf.logger.info(f"Running epoch: {epoch}/{hyperparams.epochs}")
        for data in train_loader:
            loss, out_nodes, out_edges = train_batch(data, model, optimizer, criterion)

            batch_counter += 1
            example_counter += len(data)
            if (batch_counter % 25) == 0:
                exp_out_nodes = torch.argmax(data.y, dim=1)
                exp_out_edges = torch.argmax(data.edge_output_attr, dim=1)

                out_nodes = torch.argmax(torch.exp(out_nodes), dim=1)
                out_edges = torch.argmax(torch.exp(out_edges), dim=1)

                accuracy_nodes = accuracy(exp_out_nodes, out_nodes)
                accuracy_edges = accuracy(exp_out_edges, out_edges)

                wandb.log({"loss": loss,
                           "accuracy_nodes": accuracy_nodes,
                           "accuracy_edges": accuracy_edges}, step=example_counter)

                if accuracy_edges > best_accuracy:
                    conf.logger.info(f"Saving model with accuracy: {accuracy_edges}")
                    torch.save(model.state_dict(), conf.model_path)

def train_batch(data, model, optimizer, criterion):
    out_nodes, out_edges = model(data)

    y, edge_output_attr = torch.argmax(data.y, dim=1), torch.argmax(data.edge_output_attr, dim=1)
    loss_nodes = criterion(out_nodes, y)
    loss_edges = criterion(out_edges, edge_output_attr)

    optimizer.zero_grad()
    # loss_nodes.backward(retain_graph=True)
    loss_edges.backward()

    optimizer.step()

    return loss_edges, out_nodes, out_edges


def test(model, test_loader, conf):
    model.eval()

    conf.logger.info("Testing trained neural network.")
    with torch.no_grad():
        for data in tqdm(test_loader, disable=conf.tqdm_disable):
            out_nodes, out_edges = model(data)
            out_nodes, out_edges = torch.argmax(torch.exp(out_nodes), dim=1), torch.argmax(torch.exp(out_edges), dim=1)
            visualize_output_image(data, out_nodes, out_edges, conf.visualize_path)
            # visualize_input_image(data, "/home/lpiwowar/master-thesis/train/input_images_test")


def train_pipeline(conf):
    hyperparameters = dict(
        learning_rage=conf.learning_rate,
        batch_size=conf.batch_size,
        epochs=conf.epochs
    )

    wandb_params = dict(
        project="table-recognition",
        name=datetime.datetime.now().strftime("%Y-%m-%d-%H:%M"),
        mode=conf.wandb_mode,
        config=hyperparameters
    )

    with wandb.init(**wandb_params):
        hyperparams = wandb.config
        model, train_loader, test_loader, criterion, optimizer = make(hyperparams, conf)
        train(model, train_loader, test_loader, criterion, optimizer, hyperparams, conf)
        test(model, test_loader, conf)

"""
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


    #wandb.init(project="table-recognition",
    #           name=datetime.datetime.now().strftime("%Y-%m-%d-%H:%M"),
    #           entity="lpiwowar",
    #           mode=conf.wandb_mode,
    #           config=wandb_config)
    #wandb.watch(model)

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
"""
