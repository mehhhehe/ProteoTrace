from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score

from data_loader import load_raw_ogbn_proteins


def collect_sensitivity_results(
    root: str,
    model_dir: str,
    runs_subdir: str = "sensitivity_runs",
) -> List[Dict[str, Any]]:
    """
    Inspect saved sensitivity run directories and compute AUC metrics
    for both GraphSAGE and the GraphSAGE+XGB hybrid (if available).

    Expected directory structure (per param/value):

        model_dir/
          sensitivity_runs/
            <param>/
              <value>/
                graphsage_probs_val.npy
                graphsage_probs_test.npy
                graphsage_xgb_probs_val.npy   (optional)
                graphsage_xgb_probs_test.npy  (optional)

    Parameters
    ----------
    root : str
        Root directory containing ogbn-proteins/raw (for labels and splits).
    model_dir : str
        Directory holding the sensitivity_runs/ tree.
    runs_subdir : str, optional
        Name of the subdirectory under model_dir that stores sensitivity models.

    Returns
    -------
    results : list of dict
        One entry per (param, value) configuration with fields:
        - analysis
        - param
        - param_value
        - gnn_val_auc, gnn_test_auc
        - hybrid_xgb_val_auc, hybrid_xgb_test_auc (may be None)
        - hybrid_xgb_available (bool)
        - n_labels_used
    """
    # Load labels and split indices once
    _, labels, split_idx = load_raw_ogbn_proteins(root)
    val_idx = split_idx["valid"]
    test_idx = split_idx["test"]

    runs_root = os.path.join(model_dir, runs_subdir)
    if not os.path.isdir(runs_root):
        raise FileNotFoundError(
            f"No sensitivity runs directory found at {runs_root}. "
            "Expected model_dir/sensitivity_runs/<param>/<value>/..."
        )

    results: List[Dict[str, Any]] = []

    for param_name in sorted(os.listdir(runs_root)):
        param_dir = os.path.join(runs_root, param_name)
        if not os.path.isdir(param_dir):
            continue

        # Sort values numerically if possible
        def _value_key(x: str):
            try:
                return float(x)
            except ValueError:
                return x

        for value_name in sorted(os.listdir(param_dir), key=_value_key):
            cfg_dir = os.path.join(param_dir, value_name)
            if not os.path.isdir(cfg_dir):
                continue

            try:
                param_value: Any = int(value_name)
            except ValueError:
                try:
                    param_value = float(value_name)
                except ValueError:
                    param_value = value_name

            print(f"[Collect] Processing {param_name}={value_name} in {cfg_dir}")

            # --- GNN metrics ---
            gnn_val_path = os.path.join(cfg_dir, "graphsage_probs_val.npy")
            gnn_test_path = os.path.join(cfg_dir, "graphsage_probs_test.npy")

            if not (os.path.exists(gnn_val_path) and os.path.exists(gnn_test_path)):
                print(f"  [WARN] Missing GraphSAGE prob files in {cfg_dir}, skipping.")
                continue

            val_probs_gnn = np.load(gnn_val_path)
            test_probs_gnn = np.load(gnn_test_path)

            gnn_val_auc = float(
                roc_auc_score(labels[val_idx], val_probs_gnn, average="macro")
            )
            gnn_test_auc = float(
                roc_auc_score(labels[test_idx], test_probs_gnn, average="macro")
            )

            # --- Hybrid metrics (optional) ---
            hyb_val_path = os.path.join(cfg_dir, "graphsage_xgb_probs_val.npy")
            hyb_test_path = os.path.join(cfg_dir, "graphsage_xgb_probs_test.npy")

            hybrid_xgb_available = os.path.exists(hyb_val_path) and os.path.exists(
                hyb_test_path
            )

            if hybrid_xgb_available:
                val_probs_hyb = np.load(hyb_val_path)
                test_probs_hyb = np.load(hyb_test_path)
                hybrid_val_auc = float(
                    roc_auc_score(labels[val_idx], val_probs_hyb, average="macro")
                )
                hybrid_test_auc = float(
                    roc_auc_score(labels[test_idx], test_probs_hyb, average="macro")
                )
            else:
                hybrid_val_auc = None
                hybrid_test_auc = None

            results.append(
                {
                    "analysis": "gnn_hybrid_sensitivity_from_saved",
                    "param": param_name,
                    "param_value": param_value,
                    "gnn_val_auc": gnn_val_auc,
                    "gnn_test_auc": gnn_test_auc,
                    "hybrid_xgb_val_auc": hybrid_val_auc,
                    "hybrid_xgb_test_auc": hybrid_test_auc,
                    "hybrid_xgb_available": hybrid_xgb_available,
                    "n_labels_used": int(labels.shape[1]),
                }
            )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect sensitivity metrics from saved GraphSAGE and hybrid models "
            "under model_dir/sensitivity_runs and write sensitivity.json."
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory containing ogbn-proteins/raw (for labels/splits).",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing sensitivity_runs/ subdirectory.",
    )
    parser.add_argument(
        "--runs_subdir",
        type=str,
        default="sensitivity_runs",
        help="Name of the subdirectory under model_dir with sensitivity runs.",
    )

    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    results = collect_sensitivity_results(
        root=args.root,
        model_dir=args.model_dir,
        runs_subdir=args.runs_subdir,
    )

    out_path = os.path.join(args.model_dir, "sensitivity.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Sensitivity summary written to {out_path}")


if __name__ == "__main__":
    main()
