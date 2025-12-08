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


def run_gnn_sensitivity(
    root: str,
    model_dir: str,
    agg_method: str,
    add_degree: bool,
    param: str,
    values: List[int],
    hidden_dim: int,
    num_layers: int,
    num_neighbors: List[int],
    batch_size: int,
    epochs: int,
    device: torch.device,
) -> List[Dict]:
    """
    Run a hyperparameter sensitivity analysis for the GraphSAGE model.

    For each value in ``values`` we re-train GraphSAGE with that parameter
    changed and record validation / test ROC-AUC. All other hyperparameters
    are kept fixed.
    """
    graph, labels, split_idx = load_raw_ogbn_proteins(root)
    features = aggregate_edge_features(graph, method=agg_method, add_degree=add_degree)

    train_idx = split_idx["train"]
    val_idx = split_idx["valid"]
    test_idx = split_idx["test"]

    results: List[Dict] = []

    for v in values:
        # Copy the baseline configuration
        hdim = hidden_dim
        nlayers = num_layers
        nneigh = list(num_neighbors)
        bs = batch_size
        nepochs = epochs

        if param == "hidden_dim":
            hdim = int(v)
        elif param == "num_layers":
            nlayers = int(v)
        elif param == "num_neighbors":
            # same neighbour count for each layer
            nneigh = [int(v)] * nlayers
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

        _, _, _, val_auc, test_auc = train_gnn(
            graph=graph,
            features=features,
            labels=labels,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            hidden_dim=hdim,
            num_layers=nlayers,
            num_neighbors=nneigh,
            batch_size=bs,
            epochs=nepochs,
            model_dir=cfg_model_dir,
            device=device,
        )

        result = {
            "analysis": "gnn_hyperparam_sensitivity",
            "param": param,
            "param_value": v,
            "val_auc": float(val_auc),
            "test_auc": float(test_auc),
            # record full config used
            "hidden_dim": int(hdim),
            "num_layers": int(nlayers),
            "num_neighbors": [int(x) for x in nneigh],
            "batch_size": int(bs),
            "epochs": int(nepochs),
        }
        results.append(result)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hyperparameter sensitivity analysis for GraphSAGE on ogbn-proteins."
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
        choices=["hidden_dim", "num_layers", "num_neighbors", "batch_size", "epochs"],
        help="Hyperparameter to vary.",
    )
    parser.add_argument(
        "--values",
        type=int,
        nargs="+",
        required=True,
        help="List of values to evaluate for the chosen parameter.",
    )

    # Baseline configuration (around which we vary one parameter)
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Baseline hidden dimension for GraphSAGE.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="Baseline number of GraphSAGE layers.",
    )
    parser.add_argument(
        "--num_neighbors",
        type=int,
        nargs="+",
        default=[25, 15, 10],
        help="Baseline neighbour sampling sizes.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Baseline batch size for neighbour sampling.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Baseline number of training epochs.",
    )

    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    device = torch.device("cpu")

    add_degree = not args.no_add_degree

    results = run_gnn_sensitivity(
        root=args.root,
        model_dir=args.model_dir,
        agg_method=args.agg_method,
        add_degree=add_degree,
        param=args.param,
        values=args.values,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
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
