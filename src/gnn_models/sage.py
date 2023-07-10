import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg


class SAGE(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        hidden_dim: int,
        output_dim: int,
        feature_pre: bool = True,
        num_embeddings: int | None = None,
        num_layers: int = 3,
        dropout: bool = True,
    ):
        super().__init__()
        self.feature_pre = feature_pre
        self.num_layers = num_layers
        self.dropout = dropout
        if feature_pre:
            if num_embeddings is not None:
                self.pre = nn.Embedding(num_embeddings, feature_dim)
            else:
                self.pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = tg.nn.SAGEConv(feature_dim, hidden_dim)
        else:
            self.conv_first = tg.nn.SAGEConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList(
            [tg.nn.SAGEConv(hidden_dim, hidden_dim) for i in range(num_layers - 2)]
        )
        self.conv_out = tg.nn.SAGEConv(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        if self.feature_pre:
            x = self.pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.num_layers - 2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x
