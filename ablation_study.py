# ablation_study.py

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from data_loader import load_raw_ogbn_proteins, aggregate_edge_features
from train import train_baseline, train_gnn
from shap_analysis import build_gat_encoder, compute_embeddings  # reuse


from train_hybrid import train_hybrid_classifier  # from the earlier file


def run_baseline_config(
    graph: Dict,
    labels: np.ndarray,
    split_idx: Dict[str, np.ndarray],
    agg_method: str,
    add_degree: bool,
    out_dir: str,
) -> Dict:
    features = aggregate_edge_features(graph, method=agg_method, add_degree=add_degree)
    model_dir = out_dir
    _, _, _, val_auc, test_auc = train_baseline(
        features,
        labels,
        split_idx["train"],
        split_idx["valid"],
        split_idx["test"],
        model_dir,
    )
    return {
        "model_family": "baseline",
        "agg_method": agg_method,
        "add_degree": add_degree,
        "val_auc": float(val_auc),
        "test_auc": float(test_auc),
    }


def run_gnn_and_hybrid_config(
    graph: Dict,
    labels: np.ndarray,
    split_idx: Dict[str, np.ndarray],
    agg_method: str,
    add_degree: bool,
    hidden_dim: int,
    num_layers: int,
    heads: int,
    num_neighbors: List[int],
    batch_size: int,
    epochs: int,
    out_dir: str,
    device: torch.device,
) -> Dict:
    features = aggregate_edge_features(graph, method=agg_method, add_degree=add_degree)

    # 1) Pure GNN
    train_probs_gnn, val_probs_gnn, test_probs_gnn, val_auc_gnn, test_auc_gnn = train_gnn(
        graph,
        features,
        labels,
        split_idx["train"],
        split_idx["valid"],
        split_idx["test"],
        hidden_dim,
        num_layers,
        heads,
        num_neighbors,
        batch_size,
        epochs,
        out_dir,
        device,
    )

    result = {
        "model_family": "gnn",
        "agg_method": agg_method,
        "add_degree": add_degree,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_neighbors": num_neighbors,
        "batch_size": batch_size,
        "epochs": epochs,
        "gnn_val_auc": float(val_auc_gnn),
        "gnn_test_auc": float(test_auc_gnn),
    }

    # 2) Hybrid GNN + XGB (if xgboost is available)
    try:
        encoder = build_gat_encoder(
            input_dim=features.shape[1],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            model_dir=out_dir,
            device=device,
        )
        embeddings = compute_embeddings(encoder, features, graph, device=device)
        train_idx = split_idx["train"]
        val_idx = split_idx["valid"]
        test_idx = split_idx["test"]

        val_auc_h, test_auc_h = train_hybrid_classifier(
            embeddings,
            labels,
            train_idx,
            val_idx,
            test_idx,
            classifier="xgb",
            model_dir=out_dir,
        )
        result["hybrid_xgb_val_auc"] = float(val_auc_h)
        result["hybrid_xgb_test_auc"] = float(test_auc_h)
    except ImportError as e:
        result["hybrid_xgb_error"] = str(e)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation study for ogbn-proteins models")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_neighbors", type=int, nargs="+", default=[15, 15, 10])
    parser.add_argument("--heads", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu")

    graph, labels, split_idx = load_raw_ogbn_proteins(args.root)

    results: List[Dict] = []

    # 1) Baseline ablations
    baseline_cfgs = [
        ("baseline_mean_deg", "mean", True),
        ("baseline_mean_nodeg", "mean", False),
        ("baseline_sum_deg", "sum", True),
    ]
    for name, agg_method, add_degree in baseline_cfgs:
        cfg_dir = os.path.join(args.out_dir, name)
        os.makedirs(cfg_dir, exist_ok=True)
        print(f"[Ablation] Running baseline config {name}")
        res = run_baseline_config(
            graph,
            labels,
            split_idx,
            agg_method=agg_method,
            add_degree=add_degree,
            out_dir=cfg_dir,
        )
        res["config_name"] = name
        results.append(res)

    # 2) GNN + Hybrid ablations
    gnn_cfgs = [
        ("gnn_small", "mean", True, 64, 2),
        ("gnn_deep", "mean", True, 128, 3),
        ("gnn_wide", "mean", True, 256, 3),
    ]
    for name, agg_method, add_degree, hidden_dim, num_layers in gnn_cfgs:
        cfg_dir = os.path.join(args.out_dir, name)
        os.makedirs(cfg_dir, exist_ok=True)
        print(f"[Ablation] Running GNN+Hybrid config {name}")
        res = run_gnn_and_hybrid_config(
            graph,
            labels,
            split_idx,
            agg_method=agg_method,
            add_degree=add_degree,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=args.heads,
            num_neighbors=args.num_neighbors,
            batch_size=args.batch_size,
            epochs=args.epochs,
            out_dir=cfg_dir,
            device=device,
        )
        res["config_name"] = name
        results.append(res)

    # Save all ablation results
    out_path = os.path.join(args.out_dir, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Ablation results saved to {out_path}")


if __name__ == "__main__":
    main()
