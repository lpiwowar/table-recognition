import numpy as np
import torch
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn.meta import MetaLayer


class SimpleModelWithNNEnding(torch.nn.Module):
    def __init__(self, num_node_features=4, num_edge_features=6, num_edge_classes=4, num_node_classes=2):
        super().__init__()
        self.conv1 = GATConv(num_node_features, 16, edge_dim=num_edge_features)
        self.edge1 = torch.nn.Sequential(
            torch.nn.Linear((16 * 2) + num_edge_features, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(32, 16),
            # torch.nn.ReLU(),
        )
        self.conv2 = GATConv(16, 16, edge_dim=16)

        self.nodeLinear = torch.nn.Sequential(
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(16, num_node_classes),
            torch.nn.ReLU(),
        )

        self.edgeLinear = torch.nn.Sequential(
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(16, num_edge_classes),
            torch.nn.ReLU(),
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_attr)
        edge_attr = self.edge1(torch.cat((x[edge_index[0]], edge_attr, x[edge_index[1]]), dim=1).float())
        x = self.conv2(x, edge_index, edge_attr)

        x = self.nodeLinear(x)
        edge_attr = self.edgeLinear(edge_attr)

        # Nodes, Edges
        # Odstranit softmaxy - nejprve rozsir informaci -> pak transformuj
        # Positional embbeding
        # Hrany: Smer, nejblizsi bod (ne stredy) -- Nenacpat jako positional ambbeding?
        return F.log_softmax(x, dim=1), F.log_softmax(edge_attr, dim=1)


class EdgeSubModel(torch.nn.Module):
    def __init__(self, in_node_features, in_edge_features, hidden_features, out_edge_features, residual):
        super().__init__()
        self.residual = residual
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * in_node_features + in_edge_features, hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features, out_edge_features)
        )

    def forward(self, src_node_features, dest_node_features, edge_attr, u, batch):
        out = torch.cat([src_node_features, dest_node_features, edge_attr], 1)
        out = self.edge_mlp(out)
        if self.residual:
            out = out + edge_attr

        return out


class NodeSubModel(torch.nn.Module):
    def __init__(self, in_node_features, in_edge_features, hidden_features, out_node_features, residual):
        super().__init__()
        self.residual = residual
        self.node_mlp_1 = torch.nn.Sequential(
            torch.nn.Linear(in_node_features + in_edge_features, hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features, out_node_features)
        )

        self.node_mlp_2 = torch.nn.Sequential(
            torch.nn.Linear(in_node_features + out_node_features, hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features, out_node_features)
        )

    def forward(self, src_node_features, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = torch.cat([src_node_features[col], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        #out = scatter_mean(out, row, dim=0, dim_size=src_node_features.size(0))
        #out = torch.cat([src_node_features, out], dim=1)
        #out = self.node_mlp_2(out)
        if self.residual:
            out = out + src_node_features

        return out


class NodeEdgeMLPEnding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.node_edge_layer_1 = NodeEdgeMLPEnding.get_node_edge_layer(in_node_features=4, in_edge_features=6,
                                                                       hidden_features=32, out_node_features=16,
                                                                       out_edge_features=16, residual=False)

        self.node_edge_layer_2 = NodeEdgeMLPEnding.get_node_edge_layer(in_node_features=16, in_edge_features=16,
                                                                       hidden_features=64, out_edge_features=16,
                                                                       out_node_features=16, residual=True)

        self.node_edge_layer_3 = NodeEdgeMLPEnding.get_node_edge_layer(in_node_features=16, in_edge_features=16,
                                                                       hidden_features=64, out_edge_features=16,
                                                                       out_node_features=16, residual=True)

        self.node_edge_layer_4 = NodeEdgeMLPEnding.get_node_edge_layer(in_node_features=16, in_edge_features=16,
                                                                       hidden_features=64, out_edge_features=16,
                                                                       out_node_features=16, residual=True)

        self.node_classifier = torch.nn.Sequential(
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(16, 2),
            torch.nn.ReLU(),
        )

        self.edge_classifier = torch.nn.Sequential(
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(16, 4),
            torch.nn.ReLU(),
        )

    @staticmethod
    def get_node_edge_layer(in_node_features, in_edge_features, hidden_features, out_node_features,
                            out_edge_features, residual):
        return MetaLayer(
            edge_model=EdgeSubModel(in_node_features=in_node_features, in_edge_features=in_edge_features,
                                    hidden_features=hidden_features, out_edge_features=out_edge_features,
                                    residual=residual),
            node_model=NodeSubModel(in_node_features=in_node_features, in_edge_features=out_edge_features,
                                    hidden_features=hidden_features, out_node_features=out_node_features,
                                    residual=residual)
        )

    def forward(self, data):
        x, edge_attr = data.x, data.edge_attr
        x, edge_attr, _ = self.node_edge_layer_1(x, data.edge_index, edge_attr=edge_attr)
        x, edge_attr, _ = self.node_edge_layer_2(x, data.edge_index, edge_attr=edge_attr)
        x, edge_attr, _ = self.node_edge_layer_3(x, data.edge_index, edge_attr=edge_attr)
        x, edge_attr, _ = self.node_edge_layer_4(x, data.edge_index, edge_attr=edge_attr)

        x = self.node_classifier(x)
        edge_attr = self.edge_classifier(edge_attr)

        return F.log_softmax(x, dim=1), F.log_softmax(edge_attr, dim=1)


"""
class EdgeModel(torch.nn.Module):
    def __init__(self, n_features, n_edge_features, hiddens, n_targets, residuals):
        super().__init__()
        self.residuals = residuals
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * n_features + n_edge_features, hiddens),
            torch.nn.ReLU(),
            torch.nn.Linear(hiddens, n_targets),
        )

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        out = torch.cat([src, dest, edge_attr], 1)
        out = self.edge_mlp(out)
        if self.residuals:
            out = out + edge_attr
        return out

"""
"""
class NodeModel(torch.nn.Module):
    def __init__(self, n_features, n_edge_features, hiddens, n_targets, residuals):
        super(NodeModel, self).__init__()
        self.residuals = residuals
        self.node_mlp_1 = torch.nn.Sequential(
            torch.nn.Linear(n_features + n_edge_features, hiddens),
            torch.nn.ReLU(),
            torch.nn.Linear(hiddens, n_targets),
        )
        self.node_mlp_2 = torch.nn.Sequential(
            torch.nn.Linear(n_features + n_targets, hiddens),
            torch.nn.ReLU(),
            torch.nn.Linear(hiddens, n_targets),
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = torch.cat([x[col], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, row, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        out = self.node_mlp_2(out)
        if self.residuals:
            out = out + x
        return out
"""
"""
def build_layer(n_features, n_edge_features, n_hiddens, n_targets_edge, n_targets_node, residuals):
    return MetaLayer(
        edge_model=EdgeModel(n_features, n_edge_features, n_hiddens, n_targets_edge, residuals),
        node_model=NodeModel(n_features, n_targets_edge, n_hiddens, n_targets_node, residuals),
    )
"""
"""
class SimpleModelWithNNEnding2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = build_layer(n_features=4, n_edge_features=6, n_hiddens=32, n_targets_edge=16, n_targets_node=16,
                                  residuals=False)
        self.layer2 = build_layer(n_features=16, n_edge_features=16, n_hiddens=64, n_targets_edge=16, n_targets_node=16,
                                  residuals=True)

        self.layer3 = build_layer(n_features=16, n_edge_features=16, n_hiddens=64, n_targets_edge=16, n_targets_node=16,
                                  residuals=True)

        self.layer4 = build_layer(n_features=16, n_edge_features=16, n_hiddens=64, n_targets_edge=16, n_targets_node=16,
                                  residuals=True)

        self.nodeLinear = torch.nn.Sequential(
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(16, 2),
            torch.nn.ReLU(),
        )

        self.edgeLinear = torch.nn.Sequential(
            torch.nn.Linear(16, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(16, 4),
            torch.nn.ReLU(),
        )

    def forward(self, data):
        x, edge_attr, _ = self.layer1(data.x, data.edge_index, edge_attr=data.edge_attr)
        x, edge_attr, _ = self.layer2(x, data.edge_index, edge_attr=edge_attr)
        x, edge_attr, _ = self.layer3(x, data.edge_index, edge_attr=edge_attr)
        x, edge_attr, _ = self.layer4(x, data.edge_index, edge_attr=edge_attr)

        x = self.nodeLinear(x)
        edge_attr = self.edgeLinear(edge_attr)

        return F.log_softmax(x, dim=1), F.log_softmax(edge_attr, dim=1)
"""
