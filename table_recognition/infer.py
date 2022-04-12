import os
import tempfile

import networkx as nx
import torch
from tqdm import tqdm

from table_recognition.graph import Graph
from table_recognition.models import SimpleModel
from table_recognition.models import NodeEdgeMLPEnding
from table_recognition.models import VisualNodeEdgeMLPEnding


class Infer(object):
    def __init__(self, config):
        self.config = config
        self.prepared_data_dir = tempfile.mkdtemp()
        self.visualize_path = tempfile.mkdtemp()

        self.available_models = {
            SimpleModel.__name__: SimpleModel,
            NodeEdgeMLPEnding.__name__: NodeEdgeMLPEnding,
            VisualNodeEdgeMLPEnding.__name__: VisualNodeEdgeMLPEnding
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.available_models[self.config.model_name]()
        self.model.load_state_dict(torch.load(self.config.weights_path, map_location=torch.device(self.device)))

        self.prepare_input()

    def prepare_input(self):
        images = os.listdir(self.config.img_path)
        images.sort()
        ocr_output_path = os.listdir(self.config.ocr_output_path)
        ocr_output_path.sort()
        self.config.prepared_data_dir = self.prepared_data_dir
        self.config.visualize_dir = self.visualize_path

        self.config.logger.info("Preparing graph representation of the input tables ...")
        self.config.logger.info(f"Storing prepared graph representations to {self.prepared_data_dir}")
        self.config.logger.info(f"Visualization of output graph stored in {self.visualize_path}")
        for image_name, ocr_output in tqdm(zip(images[5:6], ocr_output_path[5:6])):
            graph = Graph(
                config=self.config,
                ocr_output_path=os.path.join(self.config.ocr_output_path, ocr_output),
                ground_truth_path=None,
                img_path=os.path.join(self.config.img_path, image_name)
            )

            graph.initialize()
            graph.color_input()
            graph.dump_infer()

            name = image_name.split(".")[0]
            file_path = os.path.join(self.prepared_data_dir, name) + ".pt"
            data = torch.load(os.path.join(file_path))

            out_nodes, out_edges = self.model(data)

            out_nodes = torch.argmax(torch.exp(out_nodes), dim=1)
            out_edges = torch.argmax(torch.exp(out_edges), dim=1)

            id_to_node = {node.id: node for node in graph.nodes}
            id_to_edge = {(edge.node1.id, edge.node2.id): edge for edge in graph.edges}

            node_num_to_name = {0: "header", 1: "data"}
            for id1, node_type in enumerate(out_nodes):
                node_type = int(node_type.numpy())
                id_to_node[id1].type = node_num_to_name[node_type]

            edge_num_to_name = {0: "cell", 1: "horizontal", 2: "vertical", 3: "no-relationship"}
            for id1, id2, edge_type in zip(data.edge_index[0], data.edge_index[1], out_edges):
                id1 = int(id1.numpy())
                id2 = int(id2.numpy())
                edge_type = int(edge_type.numpy())
                id_to_edge[(id1, id2)].type = edge_num_to_name[edge_type]

            graph2text = Graph2Text(graph)

            graph.visualize()


class GrapOptimModel(torch.nn.Module):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self.node_positions = {}
        for node in graph.nodes:
            self.node_positions[node.id] = [torch.nn.Parameter(torch.tensor(node.x)),
                                            torch.nn.Parameter(torch.tensor(node.y))]

    def forward(self):
        horizontal_error = []
        for edge in self.graph.edges:
            node1_x = self.node_positions[edge.node1.id][0]
            node2_x = self.node_positions[edge.node2.id][0]
            horizontal_error += [node1_x - node2_x]

        horizontal_error = torch.sum(torch.tensor(horizontal_error))
        return horizontal_error ** 2


class GraphOptimizer(torch.nn.Module):
    """Custom Pytorch model for gradient optimization.
    """

    def __init__(self, graph):
        super().__init__()
        self.optimizer = torch.optim.Adam(model.parameters())

        for _ in range(10000):
            self.optimizer.zero_grad()

            output = self.forward()
            loss = self.my_loss(output)
            loss.backward()

            self.optimizer.step()

    def forward(self):
        horizontal_difference = []
        for edge in self.graph.edges:
            horizontal_difference += [[edge.node1.x, edge.node2.x]]

        horizontal_difference = torch.tensor(horizontal_difference)
        horizontal_error = torch.sum(horizontal_difference[0] - horizontal_difference[1]) ** 2

        horizontal_decimal_error = []
        for edge in self.graph.edges:
            horizontal_decimal_error += [torch.floor(edge.node1.x) - edge.node1.x,
                                         torch.floor(edge.node2.x) - edge.node2.x]
        horizontal_decimal_error = torch.sum(torch.tensor(horizontal_decimal_error)) ** 2

        return horizontal_error + horizontal_decimal_error

    def my_loss(self, output):
        return output ** 2

"""
if __name__ == "__main__":
    model = Model()
    optimizer = torch.optim.Adam(model.parameters())

    for x in range(100000):
        optimizer.zero_grad()

        output = model()
        loss = my_loss(output)
        loss.backward()
        print(f"loss: {loss}, x1: {model.x1}, x2: {model.x2}, x3: {model.x3}, x4: {model.x4}")

        optimizer.step()
"""

class Graph2Text(object):
    def __init__(self, graph):
        self.graph = graph

        self.remove_no_relationship()
        self.merge_cell()
        self.remove_symmetrical_edges()
        self.remove_transitive_edges()
        self.nodes_to_grid()

    def remove_no_relationship(self):
        self.graph.edges = [edge for edge in self.graph.edges
                            if edge.type != "no-relationship"]

    def remove_symmetrical_edges(self):
        to_keep = []
        for edge in self.graph.edges:
            # Do not keep reflexive edges
            if edge.node1.id == edge.node2.id:
                continue

            if edge.type == "horizontal":
                if edge.node2.bbox["center"][1] >= edge.node1.bbox["center"][1]:
                    if (edge.node2.id, edge.node1.id) not in to_keep:
                        to_keep += [(edge.node1.id, edge.node2.id)]
            elif edge.type == "vertical":
                if edge.node2.bbox["center"][0] >= edge.node1.bbox["center"][0]:
                    if (edge.node2.id, edge.node1.id) not in to_keep:
                        to_keep += [(edge.node1.id, edge.node2.id)]
            else:
                if (edge.node2.id, edge.node1.id) not in to_keep:
                    to_keep += [(edge.node1.id, edge.node2.id)]

        self.graph.edges = [edge for edge in self.graph.edges
                            if (edge.node1.id, edge.node2.id) in to_keep]

    def merge_cell(self):
        cell_edges = [edge for edge in self.graph.edges if edge.type == "cell"]
        # print(cell_edges)

        nodes_to_remove = []
        for cell_edge in cell_edges:
            if cell_edge.node1.id == cell_edge.node2.id:
                continue

            cell_node1 = cell_edge.node1
            cell_node2 = cell_edge.node2
            nodes_to_remove += [cell_node2.id]

            for edge in self.graph.edges:
                if edge.node1.id == cell_node2.id:
                    edge.node1 = cell_node1
                elif edge.node2.id == cell_node2.id:
                    edge.node2 = cell_node1

        for edge in cell_edges:
            if edge in self.graph.edges:
                self.graph.edges.remove(edge)

        self.graph.nodes = [node for node in self.graph.nodes
                            if node.id not in nodes_to_remove]

    def remove_transitive_edges(self):
        horizontal_edges = [edge for edge in self.graph.edges
                            if edge.type == "horizontal"]
        vertical_edges = [edge for edge in self.graph.edges
                          if edge.type == "vertical"]

        horizontal_edges_ids = list(set([(edge.node1.id, edge.node2.id) for edge in horizontal_edges]))
        vertical_edges_ids = list(set([(edge.node1.id, edge.node2.id) for edge in vertical_edges]))

        horizontal_graph = nx.DiGraph(horizontal_edges_ids)
        vertical_graph = nx.DiGraph(vertical_edges_ids)

        # print(list(nx.simple_cycles(vertical_graph)))
        horizontal_graph_list = list(nx.DiGraph(horizontal_edges_ids).edges)
        vertical_graph_list = list(nx.DiGraph(vertical_edges_ids).edges)

        reduced_horizontal_graph = list(nx.transitive_reduction(horizontal_graph).edges)
        reduced_vertical_graph = list(nx.transitive_reduction(vertical_graph).edges)

        horizontal_to_remove = set(horizontal_graph_list).difference(set(reduced_horizontal_graph))
        vertical_to_remove = set(vertical_graph_list).difference(set(reduced_vertical_graph))
        to_remove = horizontal_to_remove.union(vertical_to_remove)

        self.graph.edges = [edge for edge in self.graph.edges
                            if (edge.node1.id, edge.node2.id) not in to_remove]

    def nodes_to_grid(self):
        grap_optimizer = GraphOptimizer(self.graph)
