"""Hybrid GNN + classical training for ogbn-proteins (BioGraphFusion).

This script takes a *pre-trained* GAT model (trained via train.py),
uses it as an encoder to generate node embeddings, and then trains
classical multi-label classifiers on those embeddings:

    - Logistic Regression
    - Random Forest
    - XGBoost (optional, if installed)

Each hybrid model is evaluated on validation and test splits using
macro ROC-AUC, and probabilities are saved to disk for later use
(e.g., in the Flask dashboard).

Typical usage (after running train.py first):

    python train.py --root ./data --model_dir ./models_strong \
        --agg_method mean --hidden_dim 256 --num_layers 3 \
        --batch_size 2048 --num_neighbors 25 15 10 --epochs 30

    python train_hybrid.py --root ./data --model_dir ./models_strong \
        --agg_method mean --hidden_dim 256 --num_layers 3 \
        --classifier all
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Tuple, Dict, List

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

try:
    import xgboost as xgb  # type: ignore
except ImportError:
    xgb = None

from data_loader import load_raw_ogbn_proteins, aggregate_edge_features
from train import GAT  # reuse the same architecture / state_dict


def build_gat_encoder(
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    heads: int,
    model_dir: str,
    device: torch.device,
) -> nn.Module:
    """Load the trained GAT classifier and return it as an encoder.

    We reuse the GAT class from train.py, load the state_dict
    from `gat_classifier.pth`, and then use the conv stack as
    an encoder. The final linear layer is ignored when computing
    embeddings.
    """
    ckpt_path = os.path.join(model_dir, "gat_classifier.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Expected GAT checkpoint at {ckpt_path}. "
            "Run train.py first to train the GNN."
        )

    model = GAT(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, heads=heads)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def compute_embeddings(
    model: nn.Module,
    features: np.ndarray,
    graph: Dict[str, np.ndarray],
    device: torch.device,
) -> np.ndarray:
    """Compute node embeddings for all nodes using the trained GAT.

    We pass the full graph through the conv stack and take the output
    *before* the final linear layer as the embedding.
    """
    x = torch.from_numpy(features).float().to(device)
    edge_index = torch.from_numpy(graph["edge_index"]).long().to(device)

    # Forward through conv layers manually to stop before final classifier.
    with torch.no_grad():
        h = x
        for conv in model.convs:  # type: ignore[attr-defined]
            h = conv(h, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=model.dropout, training=False)  # type: ignore[attr-defined]

    embeddings = h.cpu().numpy()
    return embeddings


def standardise(
    Z: np.ndarray,
    train_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardise embeddings using training subset statistics."""
    mean = Z[train_idx].mean(axis=0, keepdims=True)
    std = Z[train_idx].std(axis=0, keepdims=True) + 1e-6
    return (Z - mean) / std, mean, std


def train_hybrid_classifier(
    Z: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    classifier: str,
    model_dir: str,
) -> Tuple[float, float]:
    """Train a hybrid classifier on GNN embeddings and save outputs.

    Parameters
    ----------
    Z : np.ndarray
        Node embeddings of shape [num_nodes, d].
    labels : np.ndarray
        Binary label matrix of shape [num_nodes, 112].
    train_idx, val_idx, test_idx : np.ndarray
        Indices of training, validation and test nodes.
    classifier : {"logreg", "rf", "xgb"}
        Type of classical model to train.
    model_dir : str
        Directory where models and predictions will be saved.

    Returns
    -------
    (val_auc, test_auc) : tuple of floats
    """
    os.makedirs(model_dir, exist_ok=True)

    Z_std, mean, std = standardise(Z, train_idx)
    Z_train = Z_std[train_idx]
    Z_val = Z_std[val_idx]
    Z_test = Z_std[test_idx]

    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]

    clf_name = classifier.lower()
    if clf_name == "logreg":
        base_est = LogisticRegression(
            max_iter=400,
            n_jobs=-1,
            solver="lbfgs",
        )
        clf = OneVsRestClassifier(base_est, n_jobs=-1)
        tag = "gat_logreg"
    elif clf_name == "rf":
        base_est = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
        )
        # RandomForest can handle multi-label with OneVsRest as well
        clf = OneVsRestClassifier(base_est, n_jobs=1)
        tag = "gat_rf"
    elif clf_name == "xgb":
        if xgb is None:
            raise ImportError(
                "xgboost is not installed. Install it via `pip install xgboost` "
                "to use the XGBoost hybrid."
            )
        base_est = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=-1,
            objective="binary:logistic",
        )
        clf = OneVsRestClassifier(base_est, n_jobs=1)
        tag = "gat_xgb"
    else:
        raise ValueError(f"Unsupported classifier: {classifier}")

    print(f"[Hybrid] Training {tag} on embeddings with shape {Z_train.shape}")
    clf.fit(Z_train, y_train)

    # Save the hybrid model and normalisation stats
    with open(os.path.join(model_dir, f"{tag}_classifier.pkl"), "wb") as f:
        pickle.dump(
            {
                "classifier": clf,
                "mean": mean,
                "std": std,
            },
            f,
        )

    # Probabilities
    train_probs = clf.predict_proba(Z_train)
    val_probs = clf.predict_proba(Z_val)
    test_probs = clf.predict_proba(Z_test)

    # ROC-AUC
    val_auc = roc_auc_score(y_val, val_probs, average="macro")
    test_auc = roc_auc_score(y_test, test_probs, average="macro")

    # Save probability matrices for dashboard / analysis
    np.save(os.path.join(model_dir, f"{tag}_probs_train.npy"), train_probs)
    np.save(os.path.join(model_dir, f"{tag}_probs_val.npy"), val_probs)
    np.save(os.path.join(model_dir, f"{tag}_probs_test.npy"), test_probs)

    print(f"[Hybrid] {tag}: val_auc={val_auc:.4f}, test_auc={test_auc:.4f}")
    return val_auc, test_auc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train hybrid GAT + classical models on ogbn-proteins embeddings"
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory containing ogbn-proteins/raw",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory with GAT checkpoint and where hybrid models will be saved",
    )
    parser.add_argument(
        "--agg_method",
        type=str,
        default="mean",
        choices=["mean", "sum"],
        help="Edge feature aggregation method (must match training)",
    )
    parser.add_argument(
        "--no_add_degree",
        action="store_true",
        help="Do not append degree/log-degree (must match training)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension used when training GAT (must match)",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of GAT layers used during training (must match)",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=2,
        help="Attention heads used when training the GAT (must match)",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="all",
        choices=["logreg", "rf", "xgb", "all"],
        help="Which hybrid classifier to train",
    )
    args = parser.parse_args()

    device = torch.device("cpu")

    # Load graph and labels
    graph, labels, split_idx = load_raw_ogbn_proteins(args.root)
    features = aggregate_edge_features(
        graph,
        method=args.agg_method,
        add_degree=not args.no_add_degree,
    )

    # Build encoder and compute embeddings
    encoder = build_gat_encoder(
        input_dim=features.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        model_dir=args.model_dir,
        device=device,
    )
    embeddings = compute_embeddings(encoder, features, graph, device=device)

    train_idx = split_idx["train"]
    val_idx = split_idx["valid"]
    test_idx = split_idx["test"]

    # Train requested hybrid models
    requested = args.classifier.lower()
    if requested == "all":
        for clf_name in ["logreg", "rf", "xgb"]:
            try:
                train_hybrid_classifier(
                    embeddings,
                    labels,
                    train_idx,
                    val_idx,
                    test_idx,
                    classifier=clf_name,
                    model_dir=args.model_dir,
                )
            except ImportError as e:
                print(f"[Hybrid] Skipping {clf_name}: {e}")
    else:
        train_hybrid_classifier(
            embeddings,
            labels,
            train_idx,
            val_idx,
            test_idx,
            classifier=requested,
            model_dir=args.model_dir,
        )


if __name__ == "__main__":
    main()
