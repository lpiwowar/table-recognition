import datetime

import torch
import wandb
from torch_geometric.loader import DataLoader
from torch_geometric.utils.metric import accuracy
from tqdm import tqdm

from table_recognition.dataset import TableDataset
from table_recognition.graph.utils import visualize_output_image
from table_recognition.models import SimpleModel
from table_recognition.models import NodeEdgeMLPEnding
from table_recognition.models import VisualNodeEdgeMLPEnding


class Trainer(object):
    def __init__(self, conf):
        self.conf = conf
        self.device = None
        self.model = None
        self.available_models = {
            SimpleModel.__name__: SimpleModel,
            NodeEdgeMLPEnding.__name__: NodeEdgeMLPEnding,
            VisualNodeEdgeMLPEnding.__name__: VisualNodeEdgeMLPEnding
        }

        self.train_loader = None
        self.test_loader = None
        self.validate_loader = None

        self.optimizer = None
        self.criterion = None

        self.init_resources()
        self.train_pipeline()

    def init_resources(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.available_models[self.conf.model_name]().to(self.device)
        self.conf.logger.info(f"Using {self.device} device for training.")
        self.conf.logger.info(f"Training {type(self.model).__name__} model.")

        # -- Source: https://discuss.pytorch.org/t/finding-model-size/130275 --------
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        self.conf.logger.info('model size: {:.3f}MB'.format(size_all_mb))
        # -- Source end: https://discuss.pytorch.org/t/finding-model-size/130275 ----

        if self.conf.preload_model:
            self.model.load_state_dict(torch.load(self.conf.model_path))

        table_dataset = TableDataset(self.conf)

        train_size = int(self.conf.train_percentage * len(table_dataset))
        test_size = (len(table_dataset) - train_size) // 2
        train_dataset, test_dataset = torch.utils.data.random_split(table_dataset, [train_size, test_size * 2])
        validate_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [len(test_dataset) // 2, len(test_dataset) // 2])

        self.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=1)
        self.validate_loader = DataLoader(validate_dataset, batch_size=1)

        print("TRAIN_LOADER:")
        for file_name in self.train_loader:
            print(file_name)
        print()

        print("TEST_LOADER")
        for file_name in self.test_loader:
            print(file_name)
        print()

        print("VALIDATE_LOADER:")
        for file_name in self.validate_loader:
            print(file_name)
        print()

        exit(0)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf.learning_rate)
        self.criterion = torch.nn.NLLLoss()

    def train_pipeline(self):
        hyperparameters = dict(
            learning_rate=self.conf.learning_rate,
            batch_size=self.conf.batch_size,
            epochs=self.conf.epochs
        )

        wandb_params = dict(
            project="table-recognition",
            name=datetime.datetime.now().strftime("%Y-%m-%d-%H:%M"),
            mode=self.conf.wandb_mode,
            config=hyperparameters,
        )

        with wandb.init(**wandb_params):
            self.train()
            self.test(load_model=True, visualize=True)
            self.validate(load_model=True, visualize=True)

    def train(self):
        self.conf.logger.info("Starting training ...")
        wandb.watch(self.model, self.criterion, log="all", log_freq=10)

        mini_batch_counter = 0
        epoch_best_accuracy_edges = 0
        epoch_accuracy_nodes = []
        epoch_accuracy_edges = []
        epoch_loss = []

        for epoch in range(self.conf.epochs):
            self.conf.logger.info(f"Running epoch: {epoch}/{self.conf.epochs}")
            for data in tqdm(self.train_loader, disable=self.conf.tqdm_disable):
                data.to(self.device)

                mini_batch_counter += self.conf.gpu_max_batch
                loss, out_nodes, out_edges = self.train_batch(data, mini_batch_counter=mini_batch_counter)

                epoch_loss += [float(loss)]

                # exp_out_nodes = torch.argmax(data.y, dim=1)
                exp_out_edges = torch.argmax(data.edge_output_attr, dim=1)

                # out_nodes = torch.argmax(torch.exp(out_nodes), dim=1)
                out_edges = torch.argmax(torch.exp(out_edges), dim=1) 
                
                # epoch_accuracy_nodes += [accuracy(exp_out_nodes, out_nodes)]
                epoch_accuracy_edges += [accuracy(exp_out_edges, out_edges)]

                data.cpu()
                torch.cuda.empty_cache()

            # Calculate TRAIN metrics
            # epoch_accuracy_nodes_avg = sum(epoch_accuracy_nodes) / len(epoch_accuracy_edges)
            epoch_accuracy_nodes_avg = 0
            epoch_accuracy_edges_avg = sum(epoch_accuracy_edges) / len(epoch_accuracy_edges)
            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            self.conf.logger.info(f"TRAIN DATA => "
                                  f"[accuracy nodes: {epoch_accuracy_nodes_avg}] "
                                  f"[accuracy edges: {epoch_accuracy_edges_avg}] "
                                  f"[loss: {epoch_loss}]")

            # Calculate TEST metrics
            metrics = self.test(visualize=False, load_model=False)
            metrics_validate = self.validate(visualize=False, load_model=False)

            # Log metrics to WANDB
            wandb.log({"TRAIN DATA - loss": epoch_loss,
                       "TRAIN DATA - accuracy nodes": epoch_accuracy_nodes_avg,
                       "TRAIN DATA - accuracy edges": epoch_accuracy_edges_avg,
                       "TEST DATA - accuracy nodes": metrics["accuracy_nodes"],
                       "TEST DATA - accuracy edges": metrics["accuracy_edges"],
                       "TEST DATA - loss": metrics["loss"],
                       "VALID DATA - loss": metrics_validate["loss"],
                       "VALID DATA - accuracy_edges": metrics_validate["accuracy_edges"]
                       }, step=epoch)

            # Save model if accuracy has improved
            if metrics["accuracy_edges"] > epoch_best_accuracy_edges:
                self.conf.logger.info(f"Saving model with accuracy: {metrics['accuracy_edges']}")
                epoch_best_accuracy_edges = metrics["accuracy_edges"]
                torch.save(self.model.state_dict(), self.conf.model_path)

            epoch_accuracy_nodes_avg = []
            epoch_accuracy_edges_avg = []
            epoch_loss = []

    def train_batch(self, data, evaluation=False, mini_batch_counter=0):
        out_nodes, out_edges = self.model(data)
        # y, edge_output_attr = torch.argmax(data.y, dim=1), torch.argmax(data.edge_output_attr, dim=1)
        edge_output_attr = torch.argmax(data.edge_output_attr, dim=1)

        # loss_nodes = self.criterion(out_nodes, y)
        loss_edges = self.criterion(out_edges, edge_output_attr)

        if not evaluation:
            if mini_batch_counter % self.conf.batch_size == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            else:
                loss_edges.backward()

            # self.optimizer.zero_grad()
            # loss_nodes.backward(retain_graph=True)
            # loss_edges.backward()
            # self.optimizer.step()

        return loss_edges, None, out_edges

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
                data.to(self.device)
                counter += 1
                # self.conf.logger.info(f"Tested {counter}/{len(self.test_loader)} images ...")

                loss, out_nodes, out_edges = self.train_batch(data.to(self.device), evaluation=True)
                epoch_loss += [float(loss.item())]

                # exp_out_nodes = torch.argmax(data.y, dim=1)
                exp_out_edges = torch.argmax(data.edge_output_attr, dim=1)

                # out_nodes = torch.argmax(torch.exp(out_nodes), dim=1)
                out_edges = torch.argmax(torch.exp(out_edges), dim=1)

                # epoch_accuracy_nodes += [accuracy(exp_out_nodes, out_nodes)]
                epoch_accuracy_edges += [accuracy(exp_out_edges, out_edges)]
                
                # print(torch.cuda.memory_summary(device=self.device, abbreviated=False))
                data.cpu()
                torch.cuda.empty_cache()
                if visualize:
                    visualize_output_image(data, out_nodes, out_edges, self.conf.visualize_path)
                #    visualize_input_image(data, "/home/lpiwowar/master-thesis/train/input_images_test")

            # accuracy_nodes = sum(epoch_accuracy_nodes) / len(epoch_accuracy_nodes)
            accuracy_nodes = 0
            accuracy_edges = sum(epoch_accuracy_edges) / len(epoch_accuracy_edges)
            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            self.conf.logger.info(f"TEST DATA => "
                                  f"[accuracy nodes: {accuracy_nodes}] "
                                  f"[accuracy edges: {accuracy_edges}] "
                                  f"[loss: {epoch_loss}]")

            self.model.train()
            return {
                "accuracy_nodes": accuracy_nodes,
                "accuracy_edges": accuracy_edges,
                "loss": loss
            }

    def validate(self, load_model=False, visualize=False):
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
            for data in tqdm(self.validate_loader, disable=self.conf.tqdm_disable):
                data.to(self.device)
                counter += 1
                # self.conf.logger.info(f"Tested {counter}/{len(self.test_loader)} images ...")

                loss, out_nodes, out_edges = self.train_batch(data.to(self.device), evaluation=True)
                epoch_loss += [float(loss.item())]

                # exp_out_nodes = torch.argmax(data.y, dim=1)
                exp_out_edges = torch.argmax(data.edge_output_attr, dim=1)

                # out_nodes = torch.argmax(torch.exp(out_nodes), dim=1)
                out_edges = torch.argmax(torch.exp(out_edges), dim=1)

                # epoch_accuracy_nodes += [accuracy(exp_out_nodes, out_nodes)]
                epoch_accuracy_edges += [accuracy(exp_out_edges, out_edges)]

                # print(torch.cuda.memory_summary(device=self.device, abbreviated=False))
                data.cpu()
                torch.cuda.empty_cache()
                if visualize:
                    visualize_output_image(data, out_nodes, out_edges, "./glosat_valid_visualization")
                #    visualize_input_image(data, "/home/lpiwowar/master-thesis/train/input_images_test")

            # accuracy_nodes = sum(epoch_accuracy_nodes) / len(epoch_accuracy_nodes)
            accuracy_nodes = 0
            accuracy_edges = sum(epoch_accuracy_edges) / len(epoch_accuracy_edges)
            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            self.conf.logger.info(f"TEST DATA => "
                                  f"[accuracy nodes: {accuracy_nodes}] "
                                  f"[accuracy edges: {accuracy_edges}] "
                                  f"[loss: {epoch_loss}]")

            self.model.train()
            return {
                "accuracy_nodes": accuracy_nodes,
                "accuracy_edges": accuracy_edges,
                "loss": loss
            }
