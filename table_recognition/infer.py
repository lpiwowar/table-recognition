import os
import tempfile

import torch
from tqdm import tqdm
from scipy.spatial import ConvexHull

from table_recognition.dataset import TableDataset
from table_recognition.graph.utils import visualize_output_image
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
        for image_name, ocr_output in tqdm(zip(images[:10], ocr_output_path[:10])):
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


class Graph2Text(object):
    def __init__(self, graph):
        self.graph = graph

        self.remove_no_relationship()
        self.remove_symmetrical_edges()
        self.merge_cell()

    def remove_no_relationship(self):
        self.graph.edges = [edge for edge in self.graph.edges
                            if edge.type != "no-relationship"]
        # for edge in self.graph.edges:
        #     if edge.type == "no-relationship":
        #         self.graph.edges.remove(edge)

    def remove_symmetrical_edges(self):
        to_remove = []
        for edge in self.graph.edges:
            if (edge.node1.id, edge.node2.id) not in to_remove and \
               (edge.node2.id, edge.node1.id) not in to_remove:
                to_remove += [(edge.node1.id, edge.node2.id)]

        self.graph.edges = [edge for edge in self.graph.edges
                            if (edge.node1.id, edge.node2.id) in to_remove]
        # for edge in self.graph.edges:
        #     if (edge.node1.id, edge.node2.id) in to_remove:
        #         self.graph.edges.remove(edge)

    def merge_cell(self):
        cell_edges = [edge for edge in self.graph.edges if edge.type == "cell"]
        # Remove cell edges from the graph
        # for edge in cell_edges:
        #     if edge in self.graph.edges:
        #         self.graph.edges.remove(edge)

        nodes_to_remove = []
        for cell_edge in cell_edges:
            cell_node1 = cell_edge.node1
            cell_node2 = cell_edge.node2
            nodes_to_remove += [cell_node2.id]

            for edge in self.graph.edges:
                if edge.node1.id == cell_node2.id:
                    # node1_points = edge.node1.polygon_pts
                    # cell_node1 = cell_node1
                    # pts = node1_points + cell_node1
                    # hull = ConvexHull(pts)

                    edge.node1 = cell_node1
                    # edge.node1.polygon_pts = hull
                elif edge.node2.id == cell_node2.id:
                    # node2_points = edge.node2.polygon_pts
                    # cell_node2 = cell_node2.polygon_pts
                    # pts = node2_points + cell_node2
                    # hull = ConvexHull(pts)

                    edge.node2 = cell_node1
                    # edge.node2.polygon_pts = hull

            # if node2 in self.graph.nodes:
            #     self.graph.nodes.remove(node2)

            # self.graph.edges.remove(cell_edge)

        for edge in cell_edges:
             if edge in self.graph.edges:
                 self.graph.edges.remove(edge)

        print(nodes_to_remove)
        self.graph.nodes = [node for node in self.graph.nodes
                            if node.id not in nodes_to_remove]
        # for node in nodes_to_remove:
        #     if node in self.graph.nodes:
        #       self.graph.nodes.remove(node)

        # Remove reflexive cell edges
        # for cell_edge in cell_edges:
        #     for edge in cell_edges:
        #         if cell_edge.node1
        """
        node1 = cell_edge.node1
        node2 = cell_edge.node2

        node2_edges = [edge for edge in self.graph.edges
                       if edge.node1.id == node2.id and edge.type != "cell"]
        for edge in node2_edges:
            edge.node1.id = node1.id

        if node2 in self.graph.nodes:
            self.graph.nodes.remove(node2)

        if cell_edge in self.graph.edges:
            self.graph.edges.remove(cell_edge)

        cell_edges = [edge for edge in cell_edges if edge.node2.id == node2.id]
        """