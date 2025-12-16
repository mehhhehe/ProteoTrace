"""Neural network architectures for the ogbn‑proteins application.

This module defines a lightweight **baseline MLP** and two graph neural
networks—**GraphSAGE** and **Graph Attention Network (GAT)**—for
multi‑label classification on the ogbn‑proteins dataset.  These
architectures mirror those used in the report but are simplified to
run on CPU‑only machines with limited memory.  All models output 112
logits corresponding to protein function labels.

References:

* Hamilton et al., 2017 — GraphSAGE for inductive representation
  learning【565794615307797†L20-L26】.
* Veličković et al., 2018 — Graph Attention Networks【61684600892784†L49-L59】.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv, GATConv
except ImportError:
    SAGEConv = None  # type: ignore
    GATConv = None   # type: ignore


class MLP(nn.Module):
    """A simple multi‑layer perceptron baseline.

    The MLP operates on aggregated node features and ignores the
    underlying graph structure.  It consists of two hidden layers with
    ReLU activations and dropout.  The final layer outputs 112 logits.

    Parameters
    ----------
    in_channels : int
        Dimensionality of the input node features.
    hidden_dim : int, default=64
        Number of units in each hidden layer.
    out_channels : int, default=112
        Number of output labels.
    dropout : float, default=0.2
        Dropout probability applied after each hidden layer.
    """

    def __init__(self, in_channels: int, hidden_dim: int = 64,
                 out_channels: int = 112, dropout: float = 0.2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out(x)


class GraphSAGE(nn.Module):
    """Lightweight GraphSAGE model for ogbn‑proteins.

    This implementation uses mean aggregation and two layers by
    default.  A final linear layer maps node embeddings to 112 logits.

    Parameters
    ----------
    in_channels : int
        Dimensionality of input node features.
    hidden_dim : int, default=64
        Hidden dimension for the GraphSAGE layers.
    num_layers : int, default=2
        Number of GraphSAGE layers.  Increasing this may improve
        accuracy but also increases memory usage.
    out_channels : int, default=112
        Number of output labels.
    dropout : float, default=0.2
        Dropout probability applied after each hidden layer.
    """

    def __init__(self, in_channels: int, hidden_dim: int = 64,
                 num_layers: int = 2, out_channels: int = 112,
                 dropout: float = 0.2) -> None:
        super().__init__()
        if SAGEConv is None:
            raise ImportError("torch_geometric is required for GraphSAGE")
        self.convs = nn.ModuleList()
        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_dim))
        # Intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        # Last layer
        self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.dropout = dropout
        self.out_lin = nn.Linear(hidden_dim, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out_lin(x)


class GAT(nn.Module):
    """Lightweight Graph Attention Network.

    We use two GATConv layers with a small number of heads to reduce
    memory usage on CPU.  The hidden dimension refers to the size per
    head; the total dimension after each layer is ``hidden_dim * heads``.

    Parameters
    ----------
    in_channels : int
        Dimensionality of input node features.
    hidden_dim : int, default=16
        Hidden dimension per attention head.
    num_layers : int, default=2
        Number of attention layers.
    heads : int, default=2
        Number of attention heads per layer.  At the final layer the
        number of heads is fixed to 1.
    out_channels : int, default=112
        Number of output labels.
    dropout : float, default=0.2
        Dropout probability applied on features and attention.
    """

    def __init__(self, in_channels: int, hidden_dim: int = 16,
                 num_layers: int = 2, heads: int = 2,
                 out_channels: int = 112, dropout: float = 0.2) -> None:
        super().__init__()
        if GATConv is None:
            raise ImportError("torch_geometric is required for GAT")
        self.convs = nn.ModuleList()
        self.dropout = dropout
        # First layer
        self.convs.append(GATConv(in_channels, hidden_dim, heads=heads,
                                  dropout=dropout))
        # Intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim,
                                      heads=heads, dropout=dropout))
        # Last GAT layer with single head
        self.convs.append(GATConv(hidden_dim * heads, hidden_dim,
                                  heads=1, dropout=dropout))
        self.out_lin = nn.Linear(hidden_dim, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out_lin(x)
