import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn.meta import MetaLayer

# from table_recognition.models.node_edge_mlp_ending import NodeSubModel
# from table_recognition.models.node_edge_mlp_ending import EdgeSubModel


# Source: https://github.com/pyg-team/pytorch_geometric/issues/813
# Project: https://github.com/fgerzer/gnn_acopf/
class VisualNodeEdgeMLPEnding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.node_edge_layer_1 = VisualNodeEdgeMLPEnding.get_node_edge_layer(in_node_features=256, in_edge_features=256,
                                                                             hidden_features=256, out_node_features=512,
                                                                             out_edge_features=512, residual=True)

        self.node_edge_layer_2 = VisualNodeEdgeMLPEnding.get_node_edge_layer(in_node_features=512, in_edge_features=512,
                                                                             hidden_features=512, out_edge_features=1024,
                                                                             out_node_features=1024, residual=True)

        self.node_edge_layer_3 = VisualNodeEdgeMLPEnding.get_node_edge_layer(in_node_features=1024, in_edge_features=1024,
                                                                             hidden_features=1024, out_edge_features=512,
                                                                             out_node_features=512, residual=True)

        self.node_edge_layer_4 = VisualNodeEdgeMLPEnding.get_node_edge_layer(in_node_features=512, in_edge_features=512,
                                                                             hidden_features=512, out_edge_features=256,
                                                                             out_node_features=256, residual=True)

        self.node_classifier = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(256, 2),
            torch.nn.ReLU(),
        )

        self.edge_classifier = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(256, 4),
            torch.nn.ReLU(),
        )

        self.node_cnn = torch.nn.Sequential(
            DepthwiseSeparableConv2d(3, 64, kernel_size=(3, 3)),
            DepthwiseSeparableConv2d(64, 64, kernel_size=(3, 3)),
            torch.nn.MaxPool2d((2, 2)),
            DepthwiseSeparableConv2d(64, 128, kernel_size=(3, 3)),
            torch.nn.Conv2d(128, 256, kernel_size=(3, 3)),
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

        self.edge_cnn = torch.nn.Sequential(
            DepthwiseSeparableConv2d(3, 128, kernel_size=(3, 3)),
            DepthwiseSeparableConv2d(128, 128, kernel_size=(3, 3)),
            torch.nn.MaxPool2d((2, 2)),
            DepthwiseSeparableConv2d(128, 256, kernel_size=(3, 3)),
            torch.nn.MaxPool2d((2, 2)),
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3)),
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

        self.edge_join = torch.nn.Linear(262, 256)
        self.node_join = torch.nn.Linear(260, 256)

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
        # Get node and edge geometric input features
        x, edge_attr = data.x, data.edge_attr

        # Calculate visual input features
        node_visual_features = torch.squeeze(self.node_cnn(data.node_image_regions))
        edge_visual_features = torch.squeeze(self.edge_cnn(data.edge_image_regions))

        # Join geometric and visual features
        x = torch.cat((x, node_visual_features), dim=1)
        x = self.node_join(x)
        edge_attr = torch.cat((edge_attr, edge_visual_features), dim=1)
        edge_attr = self.edge_join(edge_attr)

        # GCN
        x, edge_attr, _ = self.node_edge_layer_1(x, data.edge_index, edge_attr=edge_attr)
        x, edge_attr, _ = self.node_edge_layer_2(x, data.edge_index, edge_attr=edge_attr)
        x, edge_attr, _ = self.node_edge_layer_3(x, data.edge_index, edge_attr=edge_attr)
        x, edge_attr, _ = self.node_edge_layer_4(x, data.edge_index, edge_attr=edge_attr)

        x = self.node_classifier(x)
        edge_attr = self.edge_classifier(edge_attr)

        return F.log_softmax(x, dim=1), F.log_softmax(edge_attr, dim=1)


class DepthwiseSeparableConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.depthwise = torch.nn.Conv2d(in_channels, in_channels, kernel_size, padding=1, groups=in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        # print(f"input_shape: {x.shape}")
        x = self.depthwise(x)
        x = self.pointwise(x)
        # print(f"output_shape: {x.shape}")
        return x


class EdgeSubModel(torch.nn.Module):
    def __init__(self, in_node_features, in_edge_features, hidden_features, out_edge_features, residual):
        super().__init__()
        self.residual = residual
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * in_node_features + in_edge_features, hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features, out_edge_features)
        )

        self.residual_combine = torch.nn.Sequential(
            torch.nn.Linear(in_edge_features + out_edge_features, out_edge_features)
        )

    def forward(self, src_node_features, dest_node_features, edge_attr, u, batch):
        out = torch.cat([src_node_features, dest_node_features, edge_attr], 1)
        out = self.edge_mlp(out)
        if self.residual:
            # out = out + edge_attr
            out = torch.cat([out, edge_attr], 1)
            out = self.residual_combine(out)

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

        self.residual_combine = torch.nn.Sequential(
            torch.nn.Linear(in_node_features + out_node_features, out_node_features)
        )

    def forward(self, src_node_features, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = torch.cat([src_node_features[col], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, row, dim=0, dim_size=src_node_features.size(0))
        out = torch.cat([src_node_features, out], dim=1)
        out = self.node_mlp_2(out)
        if self.residual:
            # out = out + src_node_features
            out = torch.cat([out, src_node_features], dim=1)
            out = self.residual_combine(out)

        return out
