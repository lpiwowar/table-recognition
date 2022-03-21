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


class Trainer(object):
    def __init__(self, conf):
        self.conf = conf
        self.device = None
        self.model = None

        self.train_loader = None
        self.test_loader = None

        self.optimizer = None
        self.criterion = None

        self.init_resources()
        self.train_pipeline()

    def init_resources(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SimpleModel().to(self.device)
        self.conf.logger.info(f"Using {self.device} device for training.")
        self.conf.logger.info(f"Training {type(self.model).__name__} model.")

        if self.conf.preload_model:
            self.model.load_state_dict(torch.load(self.conf.model_path))

        table_dataset = TableDataset(self.conf)

        train_size = int(self.conf.train_percentage * len(table_dataset))
        test_size = len(table_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(table_dataset, [train_size, test_size])

        self.train_loader = DataLoader(train_dataset, batch_size=self.conf.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=1)

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = torch.nn.NLLLoss()

    def train_pipeline(self):
        hyperparameters = dict(
            learning_rage=self.conf.learning_rate,
            batch_size=self.conf.batch_size,
            epochs=self.conf.epochs
        )

        wandb_params = dict(
            project="table-recognition",
            name=datetime.datetime.now().strftime("%Y-%m-%d-%H:%M"),
            mode=self.conf.wandb_mode,
            config=hyperparameters
        )

        with wandb.init(**wandb_params):
            self.train()
            self.test(load_model=True, visualize=True)

    def train(self):
        self.conf.logger.info("Starting training ...")
        wandb.watch(self.model, self.criterion, log="all", log_freq=10)

        epoch_best_accuracy_edges = 0
        epoch_accuracy_nodes = []
        epoch_accuracy_edges = []
        epoch_loss = []

        for epoch in tqdm(range(self.conf.epochs), disable=self.conf.tqdm_disable):
            self.conf.logger.info(f"Running epoch: {epoch}/{self.conf.epochs}")
            for data in self.train_loader:
                loss, out_nodes, out_edges = self.train_batch(data)
                epoch_loss += [float(loss)]

                exp_out_nodes = torch.argmax(data.y, dim=1)
                exp_out_edges = torch.argmax(data.edge_output_attr, dim=1)

                out_nodes = torch.argmax(torch.exp(out_nodes), dim=1)
                out_edges = torch.argmax(torch.exp(out_edges), dim=1)

                epoch_accuracy_nodes += [accuracy(exp_out_nodes, out_nodes)]
                epoch_accuracy_edges += [accuracy(exp_out_edges, out_edges)]

            # Calculate TRAIN metrics
            epoch_accuracy_nodes_avg = sum(epoch_accuracy_nodes) / len(epoch_accuracy_edges)
            epoch_accuracy_edges_avg = sum(epoch_accuracy_edges) / len(epoch_accuracy_edges)
            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            self.conf.logger.info(f"TRAIN DATA => "
                                  f"[accuracy nodes: {epoch_accuracy_nodes_avg}] "
                                  f"[accuracy edges: {epoch_accuracy_edges_avg}] "
                                  f"[loss: {epoch_loss}]")

            # Calculate TEST metrics
            metrics = self.test(visualize=False, load_model=False)

            # Log metrics to WANDB
            wandb.log({"TRAIN DATA - loss": epoch_loss,
                       "TRAIN DATA - accuracy nodes": epoch_accuracy_nodes_avg,
                       "TRAIN DATA - accuracy edges": epoch_accuracy_edges_avg,
                       "TEST DATA - accuracy nodes": metrics["accuracy_nodes"],
                       "TEST DATA - accuracy edges": metrics["accuracy_edges"],
                       "TEST DATA - loss": metrics["loss"]
                       }, step=epoch)

            # Save model if accuracy has improved
            if metrics["accuracy_edges"] > epoch_best_accuracy_edges:
                self.conf.logger.info(f"Saving model with accuracy: {metrics['accuracy_edges']}")
                epoch_best_accuracy_edges = metrics["accuracy_edges"]
                torch.save(self.model.state_dict(), self.conf.model_path)

            epoch_accuracy_nodes_avg = []
            epoch_accuracy_edges_avg = []
            epoch_loss = []

    def train_batch(self, data, evaluation=False):
        # print(data.img_path)
        out_nodes, out_edges = self.model(data)
        y, edge_output_attr = torch.argmax(data.y, dim=1), torch.argmax(data.edge_output_attr, dim=1)

        loss_nodes = self.criterion(out_nodes, y)
        loss_edges = self.criterion(out_edges, edge_output_attr)

        if not evaluation:
            self.optimizer.zero_grad()
            # loss_nodes.backward(retain_graph=True)
            loss_edges.backward()
            self.optimizer.step()

        return loss_edges, out_nodes, out_edges

    def test(self, load_model=False, visualize=False):
        if load_model:
            self.conf.logger.info(f"Testing model with weights from {self.conf.model_path}.")
            self.model.load_state_dict(torch.load(self.conf.model_path))
        else:
            self.conf.logger.info("Testing neural network with current weights.")

        self.model.eval()
        with torch.no_grad():
            counter = 0
            epoch_accuracy_nodes = []
            epoch_accuracy_edges = []
            epoch_loss = []
            for data in tqdm(self.test_loader, disable=self.conf.tqdm_disable):
                counter += 1
                # conf.logger.info(f"Tested {counter}/{len(test_loader)} images ...")

                # out_nodes, out_edges = model(data)
                loss, out_nodes, out_edges = self.train_batch(data, evaluation=True)
                epoch_loss += [float(loss)]

                # out_nodes, out_edges = torch.argmax(torch.exp(out_nodes), dim=1), torch.argmax(torch.exp(out_edges), dim=1)

                exp_out_nodes = torch.argmax(data.y, dim=1)
                exp_out_edges = torch.argmax(data.edge_output_attr, dim=1)

                out_nodes = torch.argmax(torch.exp(out_nodes), dim=1)
                out_edges = torch.argmax(torch.exp(out_edges), dim=1)

                epoch_accuracy_nodes += [accuracy(exp_out_nodes, out_nodes)]
                epoch_accuracy_edges += [accuracy(exp_out_edges, out_edges)]

                if visualize:
                    visualize_output_image(data, out_nodes, out_edges, self.conf.visualize_path)
                # visualize_input_image(data, "/home/lpiwowar/master-thesis/train/input_images_test")

            accuracy_nodes = sum(epoch_accuracy_nodes) / len(epoch_accuracy_nodes)
            accuracy_edges = sum(epoch_accuracy_edges) / len(epoch_accuracy_edges)
            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            self.conf.logger.info(f"TEST DATA => "
                                  f"[accuracy nodes: {accuracy_nodes}] "
                                  f"[accuracy edges: {accuracy_edges}] "
                                  f"[loss: {epoch_loss}]")

            return {
                "accuracy_nodes": accuracy_nodes,
                "accuracy_edges": accuracy_edges,
                "loss": loss
            }
