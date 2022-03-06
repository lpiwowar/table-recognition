import numpy as np
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class SimpleModel(torch.nn.Module):
    def __init__(self, num_node_features=2, num_edge_classes=4, num_node_classes=2):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.edge1 = torch.nn.Sequential(
            torch.nn.Linear(33, num_edge_classes),
            torch.nn.ReLU()
        )
        self.conv2 = GCNConv(16, num_node_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        print(type(edge_index))

        x = self.conv1(x, edge_index)
        test = torch.cat((x[edge_index[0]], edge_attr, x[edge_index[1]]), dim=1)
        edge_attr = self.edge1(torch.cat((x[edge_index[0]], edge_attr, x[edge_index[1]]), dim=1).float())
        x = self.conv2(x, edge_index)

        # Nodes, Edges
        return F.log_softmax(x, dim=1), F.log_softmax(edge_attr, dim=1)
