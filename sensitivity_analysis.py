# sensitivity_analysis.py

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch

from data_loader import load_raw_ogbn_proteins, aggregate_edge_features
from train import train_gnn
from train_hybrid import (
    build_gat_encoder,
    compute_embeddings,
    train_hybrid_classifier,
)


def run_gnn_and_hybrid_sensitivity(
    root: str,
    model_dir: str,
    agg_method: str,
    add_degree: bool,
    param: str,
    values: List[int],
    hidden_dim: int,
    num_layers: int,
    heads: int,
    num_neighbors: List[int],
    batch_size: int,
    epochs: int,
    device: torch.device,
) -> List[Dict]:
    """
    Hyperparameter sensitivity for BOTH:
      - pure GAT (train_gnn)
      - hybrid GAT+XGBoost (train_hybrid_classifier from train_hybrid.py)

    For each value in `values`, we:
      1) Train GAT with that hyperparameter setting and save checkpoint +
         probability matrices in a per-config subdirectory.
      2) Load the GAT checkpoint as an encoder (build_gat_encoder).
      3) Compute embeddings (compute_embeddings) and train an XGBoost hybrid
         classifier on top (train_hybrid_classifier).
    """
    # Load once; reused across configs
    graph, labels, split_idx = load_raw_ogbn_proteins(root)
    features = aggregate_edge_features(graph, method=agg_method, add_degree=add_degree)

    train_idx = split_idx["train"]
    val_idx = split_idx["valid"]
    test_idx = split_idx["test"]

    results: List[Dict] = []

    for v in values:
        # Start from baseline configuration
        hdim = hidden_dim
        nlayers = num_layers
        attn_heads = heads
        nneigh = list(num_neighbors)
        bs = batch_size
        nepochs = epochs

        # Override the chosen parameter
        if param == "hidden_dim":
            hdim = int(v)
        elif param == "num_layers":
            nlayers = int(v)
            # keep num_neighbors length consistent if user only passed one value
            if len(nneigh) == 1:
                nneigh = [nneigh[0]] * nlayers
        elif param == "num_neighbors":
            # same neighbour count for each layer
            nneigh = [int(v)] * nlayers
        elif param == "heads":
            attn_heads = int(v)
        elif param == "batch_size":
            bs = int(v)
        elif param == "epochs":
            nepochs = int(v)
        else:
            raise ValueError(f"Unsupported sensitivity parameter: {param}")

        cfg_name = f"{param}={v}"
        cfg_model_dir = os.path.join(model_dir, "sensitivity_runs", param, str(v))
        os.makedirs(cfg_model_dir, exist_ok=True)

        print(
            f"[Sensitivity] Running config {cfg_name} "
            f"(hidden_dim={hdim}, layers={nlayers}, neigh={nneigh}, "
            f"batch_size={bs}, epochs={nepochs})"
        )

        # --- 1) Train GAT for this configuration ---
        (
            _train_probs_gnn,
            _val_probs_gnn,
            _test_probs_gnn,
            val_auc_gnn,
            test_auc_gnn,
        ) = train_gnn(
            graph=graph,
            features=features,
            labels=labels,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            hidden_dim=hdim,
            num_layers=nlayers,
            heads=attn_heads,
            num_neighbors=nneigh,
            batch_size=bs,
            epochs=nepochs,
            model_dir=cfg_model_dir,
            device=device,
        )

        # --- 2) Build encoder and compute embeddings for hybrid ---
        hybrid_val_auc = float("nan")
        hybrid_test_auc = float("nan")
        hybrid_error: str | None = None

        try:
            encoder = build_gat_encoder(
                input_dim=features.shape[1],
                hidden_dim=hdim,
                num_layers=nlayers,
                heads=attn_heads,
                model_dir=cfg_model_dir,
                device=device,
            )
            Z = compute_embeddings(encoder, features, graph, device=device)

            hybrid_val_auc, hybrid_test_auc = train_hybrid_classifier(
                Z,
                labels,
                train_idx,
                val_idx,
                test_idx,
                classifier="xgb",
                model_dir=cfg_model_dir,
            )
        except ImportError as e:
            # xgboost not installed
            hybrid_error = str(e)

        result = {
            "analysis": "gnn_hybrid_sensitivity",
            "param": param,
            "param_value": int(v),
            # full config used
            "hidden_dim": int(hdim),
            "num_layers": int(nlayers),
            "heads": int(attn_heads),
            "num_neighbors": [int(x) for x in nneigh],
            "batch_size": int(bs),
            "epochs": int(nepochs),
            # GNN metrics
            "gnn_val_auc": float(val_auc_gnn),
            "gnn_test_auc": float(test_auc_gnn),
            # Hybrid metrics (XGB)
            "hybrid_xgb_val_auc": float(hybrid_val_auc),
            "hybrid_xgb_test_auc": float(hybrid_test_auc),
        }
        if hybrid_error is not None:
            result["hybrid_xgb_error"] = hybrid_error

        results.append(result)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hyperparameter sensitivity for GAT and hybrid XGB on ogbn-proteins."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory containing ogbn-proteins/raw.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory where models and analysis artefacts are stored.",
    )
    parser.add_argument(
        "--agg_method",
        type=str,
        default="mean",
        choices=["mean", "sum"],
        help="Edge feature aggregation method.",
    )
    parser.add_argument(
        "--no_add_degree",
        action="store_true",
        help="Do not append degree / log-degree features to node features.",
    )

    parser.add_argument(
        "--param",
        type=str,
        default="hidden_dim",
        choices=["hidden_dim", "num_layers", "heads", "num_neighbors", "batch_size", "epochs"],
        help="Hyperparameter to vary.",
    )
    parser.add_argument(
        "--values",
        type=int,
        nargs="+",
        required=True,
        help="List of values to evaluate for the chosen parameter.",
    )

    # Baseline configuration around which we vary one parameter
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument(
        "--num_neighbors",
        type=int,
        nargs="+",
        default=[25, 10],
        help="Baseline neighbour sampling sizes.",
    )
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=20)

    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    device = torch.device("cpu")

    add_degree = not args.no_add_degree

    results = run_gnn_and_hybrid_sensitivity(
        root=args.root,
        model_dir=args.model_dir,
        agg_method=args.agg_method,
        add_degree=add_degree,
        param=args.param,
        values=args.values,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        num_neighbors=args.num_neighbors,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=device,
    )

    out_path = os.path.join(args.model_dir, f"sensitivity_results_{args.param}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Sensitivity results saved to {out_path}")


if __name__ == "__main__":
    main()
