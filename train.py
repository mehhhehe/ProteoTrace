"""Training script for the ogbn‑proteins TruthTrace‑style implementation.

This script trains both a simple non‑graph baseline and a GraphSAGE
classifier on the ogbn‑proteins dataset.  It mirrors the design of
the TruthTrace training pipeline but is adapted to run on CPU‑only
machines with limited memory.  The baseline model is a one‑vs‑rest
logistic regression classifier trained on aggregated node features.
The GNN model is a two‑layer GraphSAGE network trained with
mini‑batch neighbour sampling.  Predictions for all nodes are saved
to disk for use in the web dashboard.

Usage examples:

.. code-block:: bash

   python train.py --root ./data \
     --agg_method mean --add_degree --model_dir models \
     --epochs 5 --hidden_dim 64 --num_layers 2 \
     --batch_size 256 --num_neighbors 10 10

The above will train both the baseline and GNN on the
``data/ogbn-proteins`` dataset and write prediction files into
``models/``.
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import List, Tuple

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

from data_loader import load_raw_ogbn_proteins, aggregate_edge_features


# Save original torch.load
_original_torch_load = torch.load

def torch_load_compat(*args, **kwargs):
    # If weights_only is not specified, force the old unsafe behaviour
    # (acceptable here because the file is from a trusted source: OGB).
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

# Monkey-patch
torch.load = torch_load_compat

def train_baseline(
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    model_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Train a one‑vs‑rest logistic regression classifier as the baseline.

    Parameters
    ----------
    features : np.ndarray
        Aggregated node features of shape [num_nodes, dim].
    labels : np.ndarray
        Binary label matrix of shape [num_nodes, num_labels].
    train_idx, val_idx, test_idx : np.ndarray
        Indices of training, validation and test nodes.
    model_dir : str
        Directory where the trained classifier and predictions will be saved.

    Returns
    -------
    Tuple containing predicted probabilities for train, val and test sets
    and the validation/test ROC‑AUC scores.
    """
    os.makedirs(model_dir, exist_ok=True)
    # Standardise features
    mean = features[train_idx].mean(axis=0, keepdims=True)
    std = features[train_idx].std(axis=0, keepdims=True) + 1e-6
    X_train = (features[train_idx] - mean) / std
    X_val = (features[val_idx] - mean) / std
    X_test = (features[test_idx] - mean) / std
    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]
    # Logistic regression for multi‑label classification
    clf = OneVsRestClassifier(LogisticRegression(max_iter=200, n_jobs=-1))
    clf.fit(X_train, y_train)
    # Save the classifier
    with open(os.path.join(model_dir, "baseline_classifier.pkl"), "wb") as f:
        pickle.dump(clf, f)
    # Predict probabilities
    train_probs = clf.predict_proba(X_train)
    val_probs = clf.predict_proba(X_val)
    test_probs = clf.predict_proba(X_test)
    # Compute ROC‑AUC
    val_auc = roc_auc_score(y_val, val_probs, average="macro")
    test_auc = roc_auc_score(y_test, test_probs, average="macro")
    # Save probabilities for use in the dashboard
    np.save(os.path.join(model_dir, "baseline_probs_train.npy"), train_probs)
    np.save(os.path.join(model_dir, "baseline_probs_val.npy"), val_probs)
    np.save(os.path.join(model_dir, "baseline_probs_test.npy"), test_probs)
    return train_probs, val_probs, test_probs, val_auc, test_auc


class GraphSAGE(nn.Module):
    """Two‑layer GraphSAGE model for multi‑label classification.

    This implementation uses mean aggregation and a final linear layer
    to produce 112 logits per node.  Dropout is applied between
    layers.  The design mirrors the TruthTrace GAT model but uses
    GraphSAGE, which is lighter and suitable for CPU training.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.convs: List[SAGEConv] = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.dropout = dropout
        self.lin = nn.Linear(hidden_dim, 112)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.lin(x)
        return out


def train_gnn(
    graph: dict,
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    hidden_dim: int,
    num_layers: int,
    num_neighbors: List[int],
    batch_size: int,
    epochs: int,
    model_dir: str,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Train a GraphSAGE model with neighbour sampling and evaluate it.

    Parameters
    ----------
    graph : dict
        Raw graph dictionary from :func:`load_raw_ogbn_proteins`.
    features, labels : np.ndarray
        Aggregated node features and labels.
    train_idx, val_idx, test_idx : np.ndarray
        Node indices for training, validation and testing.
    hidden_dim : int
        Hidden dimension of the GraphSAGE layers.
    num_layers : int
        Number of GraphSAGE layers.
    num_neighbors : list[int]
        Number of sampled neighbours per layer for neighbour sampling.
    batch_size : int
        Number of root nodes per batch.
    epochs : int
        Number of training epochs.
    model_dir : str
        Directory to save model and prediction files.
    device : torch.device
        Device on which to train the model (CPU recommended).

    Returns
    -------
    Tuple containing predicted probabilities for train, val and test sets
    and the validation/test ROC‑AUC scores.
    """
	# At the start of train_gnn, after arguments:
    train_pos = {int(n): i for i, n in enumerate(train_idx)}
    val_pos   = {int(n): i for i, n in enumerate(val_idx)}
    test_pos  = {int(n): i for i, n in enumerate(test_idx)}


    os.makedirs(model_dir, exist_ok=True)
    # Build PyG Data
    x = torch.from_numpy(features).float()
    edge_index = torch.from_numpy(graph['edge_index']).long()
    y = torch.from_numpy(labels).float()
    data = Data(x=x, edge_index=edge_index, y=y)
    # Create training and evaluation masks
    num_nodes = x.size(0)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[torch.from_numpy(train_idx)] = True
    val_mask[torch.from_numpy(val_idx)] = True
    test_mask[torch.from_numpy(test_idx)] = True
    # Neighbour loaders
    train_loader = NeighborLoader(
        data,
        input_nodes=train_mask,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = NeighborLoader(
        data,
        input_nodes=val_mask,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = NeighborLoader(
        data,
        input_nodes=test_mask,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=False,
    )
    # Initialise model and optimiser
    model = GraphSAGE(input_dim=x.size(1), hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    criterion = nn.BCEWithLogitsLoss()
    best_val_auc = 0.0
    best_state_dict = None
    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index)
            # Use only root nodes (first batch.batch_size elements)
            out = logits[:batch.batch_size]
            target = batch.y[:batch.batch_size]
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * target.size(0)
            total_samples += target.size(0)
        avg_loss = total_loss / max(total_samples, 1)
        # Evaluate on validation set
        
        model.eval()
        val_probs = np.zeros((len(val_idx), labels.shape[1]), dtype=np.float32)

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index)
                probs = torch.sigmoid(logits[:batch.batch_size]).cpu().numpy()
                node_ids = batch.n_id[:batch.batch_size].cpu().numpy()

                for nid, p in zip(node_ids, probs):
                    j = val_pos[int(nid)]           # position in val_idx
                    val_probs[j] = p

        val_auc = roc_auc_score(labels[val_idx], val_probs, average="macro")
        print(f"Epoch {epoch}: train_loss={avg_loss:.4f}, val_auc={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state_dict = model.state_dict()
    # Load best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    # Compute predictions for train/val/test
    model.eval()
    

    def infer(loader: NeighborLoader, indices: np.ndarray, index_map: Dict[int, int]) -> np.ndarray:
        probs = np.zeros((len(indices), labels.shape[1]), dtype=np.float32)
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index)
                out_probs = torch.sigmoid(logits[:batch.batch_size]).cpu().numpy()
                node_ids = batch.n_id[:batch.batch_size].cpu().numpy()

                for nid, p in zip(node_ids, out_probs):
                    j = index_map[int(nid)]
                    probs[j] = p
        return probs

    train_probs = infer(train_loader, train_idx, train_pos)
    val_probs = infer(val_loader, val_idx, val_pos)
    test_probs = infer(test_loader, test_idx, test_pos)
    # Compute final metrics
    final_val_auc = roc_auc_score(labels[val_idx], val_probs, average="macro")
    final_test_auc = roc_auc_score(labels[test_idx], test_probs, average="macro")
    # Save model and probabilities
    torch.save(model.state_dict(), os.path.join(model_dir, "graphsage_classifier.pth"))
    np.save(os.path.join(model_dir, "graphsage_probs_train.npy"), train_probs)
    np.save(os.path.join(model_dir, "graphsage_probs_val.npy"), val_probs)
    np.save(os.path.join(model_dir, "graphsage_probs_test.npy"), test_probs)
    return train_probs, val_probs, test_probs, final_val_auc, final_test_auc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline and GraphSAGE models on ogbn-proteins")
    parser.add_argument("--root", type=str, required=True, help="Root directory containing ogbn-proteins/raw")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory to save models and predictions")
    parser.add_argument("--agg_method", type=str, default="mean", choices=["mean", "sum"], help="Edge feature aggregation method")
    parser.add_argument("--no_add_degree", action="store_true", help="Do not append degree and log-degree features")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension for GraphSAGE")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GraphSAGE layers")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for neighbour sampling")
    parser.add_argument("--num_neighbors", type=int, nargs='+', default=[25, 10], help="Number of sampled neighbours per GNN layer")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    args = parser.parse_args()

    # Load raw graph and labels
    graph, labels, split_idx = load_raw_ogbn_proteins(args.root)
    # Aggregate edge features to node features
    features = aggregate_edge_features(graph, method=args.agg_method, add_degree=not args.no_add_degree)
    # Train baseline classifier
    train_probs_base, val_probs_base, test_probs_base, val_auc_base, test_auc_base = train_baseline(
        features,
        labels,
        split_idx["train"],
        split_idx["valid"],
        split_idx["test"],
        args.model_dir,
    )
    print(f"Baseline ROC-AUC: val={val_auc_base:.4f}, test={test_auc_base:.4f}")
    # Train GraphSAGE classifier
    device = torch.device("cpu")
    train_probs_gnn, val_probs_gnn, test_probs_gnn, val_auc_gnn, test_auc_gnn = train_gnn(
        graph,
        features,
        labels,
        split_idx["train"],
        split_idx["valid"],
        split_idx["test"],
        args.hidden_dim,
        args.num_layers,
        args.num_neighbors,
        args.batch_size,
        args.epochs,
        args.model_dir,
        device,
    )
    print(f"GraphSAGE ROC-AUC: val={val_auc_gnn:.4f}, test={test_auc_gnn:.4f}")

    print("Training complete. Predictions saved to", args.model_dir)


if __name__ == "__main__":
    main()
