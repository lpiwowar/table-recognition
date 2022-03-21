import numpy as np
import torch
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F


class SimpleModel(torch.nn.Module):
    def __init__(self, num_node_features=4, num_edge_features=3, num_edge_classes=4, num_node_classes=2):
        super().__init__()
        self.conv1 = GATConv(num_node_features, 16, edge_dim=num_edge_features)
        self.edge1 = torch.nn.Sequential(
            torch.nn.Linear((16 * 2) + num_edge_features, 16),
            torch.nn.ReLU(),
            # torch.nn.Dropout(),
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            # torch.nn.Dropout(),
            torch.nn.Linear(8, num_edge_classes),
            # torch.nn.ReLU(),
        )
        self.conv2 = GATConv(16, num_node_classes, edge_dim=num_edge_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_attr)
        edge_attr = self.edge1(torch.cat((x[edge_index[0]], edge_attr, x[edge_index[1]]), dim=1).float())
        x = self.conv2(x, edge_index, edge_attr)

        # Nodes, Edges
        # Odstranit softmaxy - nejprve rozsir informaci -> pak transformuj
        # Positional embbeding
        # Hrany: Smer, nejblizsi bod (ne stredy) -- Nenacpat jako positional ambbeding?
        return F.log_softmax(x, dim=1), F.log_softmax(edge_attr, dim=1)
