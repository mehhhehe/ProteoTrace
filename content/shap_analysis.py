# shap_analysis.py

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torch_geometric.data import Data

from sklearn.metrics import roc_auc_score

from data_loader import load_raw_ogbn_proteins, aggregate_edge_features
from train import GraphSAGE  # reuse architecture

try:
    import shap
except ImportError:
    shap = None

try:
    import xgboost as xgb  # noqa: F401
except ImportError:
    xgb = None


def build_graphsage_encoder(
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    model_dir: str,
    device: torch.device,
) -> nn.Module:
    ckpt_path = os.path.join(model_dir, "graphsage_classifier.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"GraphSAGE checkpoint not found at {ckpt_path}. "
            "Run train.py first."
        )
    model = GraphSAGE(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
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
    x = torch.from_numpy(features).float().to(device)
    edge_index = torch.from_numpy(graph["edge_index"]).long().to(device)

    with torch.no_grad():
        h = x
        for conv in model.convs:  # type: ignore[attr-defined]
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=model.dropout, training=False)  # type: ignore[attr-defined]
    return h.cpu().numpy()


def load_hybrid_classifier(model_dir: str, tag: str):
    import pickle

    path = os.path.join(model_dir, f"{tag}_classifier.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Hybrid classifier pickle {path} not found. "
            "Run train_hybrid.py first."
        )
    with open(path, "rb") as f:
        obj = pickle.load(f)
    clf = obj["classifier"]
    mean = obj["mean"]
    std = obj["std"]
    return clf, mean, std


def shap_for_label_tree(
    base_estimator,
    X: np.ndarray,
    nsamples: int,
) -> np.ndarray:
    """Compute SHAP values for a single tree-based binary classifier."""
    if shap is None:
        raise ImportError("shap is not installed. Run `pip install shap` first.")

    # subsample to keep things tractable
    if nsamples is not None and nsamples > 0 and nsamples < X.shape[0]:
        idx = np.random.choice(X.shape[0], size=nsamples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    explainer = shap.TreeExplainer(base_estimator)
    shap_vals = explainer.shap_values(X_sample)  # shape [n_samples, d]
    return np.abs(shap_vals)  # use absolute values for importance


def run_shap_analysis(
    root: str,
    model_dir: str,
    agg_method: str,
    add_degree: bool,
    hidden_dim: int,
    num_layers: int,
    model_tag: str,
    split: str,
    nsamples: int,
    max_labels: int,
    seed: int,
) -> None:
    np.random.seed(seed)

    device = torch.device("cpu")
    graph, labels, split_idx = load_raw_ogbn_proteins(root)
    features = aggregate_edge_features(graph, method=agg_method, add_degree=add_degree)

    # Encoder
    encoder = build_graphsage_encoder(
        input_dim=features.shape[1],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        model_dir=model_dir,
        device=device,
    )
    embeddings = compute_embeddings(encoder, features, graph, device=device)

    # Which split?
    if split == "train":
        idx = split_idx["train"]
    elif split == "valid":
        idx = split_idx["valid"]
    elif split == "test":
        idx = split_idx["test"]
    else:
        raise ValueError("split must be one of: train, valid, test")

    y_split = labels[idx]
    Z_split = embeddings[idx]

    # Load hybrid classifier
    clf, mean, std = load_hybrid_classifier(model_dir, model_tag)
    Z_std = (Z_split - mean) / (std + 1e-6)

    # OneVsRestClassifier: list of estimators, one per label
    estimators = clf.estimators_

    n_labels = y_split.shape[1]
    if max_labels is not None:
        n_use = min(n_labels, max_labels)
    else:
        n_use = n_labels

    # global importance per feature across labels
    d = Z_std.shape[1]
    global_importance = np.zeros(d, dtype=np.float64)
    per_label_importance = np.zeros((n_use, d), dtype=np.float64)

    label_indices = np.arange(n_use)

    for li in label_indices:
        est = estimators[li]
        shap_vals_abs = shap_for_label_tree(est, Z_std, nsamples)
        mean_abs = shap_vals_abs.mean(axis=0)
        per_label_importance[li] = mean_abs
        global_importance += mean_abs

    global_importance /= float(n_use)

    # Save results
    out_dir = os.path.join(model_dir, "shap")
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, f"{model_tag}_global_importance.npy"), global_importance)
    np.save(os.path.join(out_dir, f"{model_tag}_per_label_importance.npy"), per_label_importance)

    summary = {
        "model_tag": model_tag,
        "split": split,
        "nsamples": nsamples,
        "max_labels": n_use,
        "feature_dim": d,
        "global_importance_top10_indices": np.argsort(global_importance)[::-1][:10].tolist(),
        "global_importance_top10_values": np.sort(global_importance)[::-1][:10].tolist(),
    }

    with open(os.path.join(out_dir, f"{model_tag}_shap_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"SHAP analysis complete for {model_tag} on {split} split.")
    print(f"Global importance (top 10 dims): {summary['global_importance_top10_indices']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SHAP analysis for hybrid GraphSAGE + tree-based models")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--agg_method", type=str, default="mean", choices=["mean", "sum"])
    parser.add_argument("--no_add_degree", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--model_tag", type=str, default="graphsage_xgb",
                        choices=["graphsage_xgb", "graphsage_rf"])
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--nsamples", type=int, default=2000, help="Number of nodes to sample for SHAP")
    parser.add_argument("--max_labels", type=int, default=20, help="Number of labels to analyse")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_shap_analysis(
        root=args.root,
        model_dir=args.model_dir,
        agg_method=args.agg_method,
        add_degree=not args.no_add_degree,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        model_tag=args.model_tag,
        split=args.split,
        nsamples=args.nsamples,
        max_labels=args.max_labels,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
