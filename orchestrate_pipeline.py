#!/usr/bin/env python
"""
orchestrate_pipeline.py

End-to-end orchestration script for the BioGraphFusion / ogbn-proteins project.

This script can:
  1. Train baseline + GAT (train.py)
  2. Train hybrid models on GNN embeddings (train_hybrid.py)
  3. Run SHAP analysis on a tree-based hybrid (shap_analysis.py)
  4. Run ablation study (ablation_study.py)
  5. Run statistical significance tests (significance_tests.py)
  6. Aggregate all analysis artefacts (aggregate_analysis.py)

Example (full pipeline):

    python orchestrate_pipeline.py \
        --root ./data \
        --model_dir ./models_main \
        --agg_method mean \
        --hidden_dim 256 \
        --num_layers 3 \
        --batch_size 2048 \
        --num_neighbors 25 15 10 \
        --epochs 30

By default all stages run. You can skip any stage with:
  --skip_train, --skip_hybrid, --skip_shap, --skip_ablation,
  --skip_significance, --skip_aggregate
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List


def run_cmd(cmd: List[str], desc: str) -> None:
    print(f"\n[ORCH] === {desc} ===")
    print("[ORCH] Command:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"{desc} failed with exit code {result.returncode}")
    print(f"[ORCH] {desc} completed.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestrate BioGraphFusion pipeline on ogbn-proteins")

    # Core paths / config
    parser.add_argument("--root", type=str, required=True,
                        help="Root directory containing ogbn-proteins/raw")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory for main models + analysis artefacts")

    parser.add_argument("--agg_method", type=str, default="mean", choices=["mean", "sum"],
                        help="Edge feature aggregation method (must be consistent across stages)")
    parser.add_argument("--no_add_degree", action="store_true",
                        help="Do not append degree/log-degree to features")

    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Hidden dimension per GAT head")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of GAT layers")
    parser.add_argument("--heads", type=int, default=2,
                        help="Attention heads for GAT")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for neighbour sampling")
    parser.add_argument("--num_neighbors", type=int, nargs="+", default=[25, 10],
                        help="Neighbours sampled per GNN layer")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Training epochs for GAT (train.py and ablation GNNs)")

    # SHAP-specific
    parser.add_argument("--shap_model_tag", type=str, default="gat_xgb",
                        choices=["gat_xgb", "gat_rf"],
                        help="Which hybrid model to analyse with SHAP")
    parser.add_argument("--shap_split", type=str, default="valid", choices=["train", "valid", "test"],
                        help="Split to use for SHAP analysis")
    parser.add_argument("--shap_nsamples", type=int, default=2000,
                        help="Number of nodes to sample for SHAP")
    parser.add_argument("--shap_max_labels", type=int, default=20,
                        help="Number of labels to include in SHAP analysis")

    # Significance-specific
    parser.add_argument("--sig_split", type=str, default="test", choices=["valid", "test"],
                        help="Split to use for significance testing")
    parser.add_argument("--sig_n_bootstrap", type=int, default=5000,
                        help="Number of bootstrap resamples for AUC difference CI")
    parser.add_argument("--sig_seed", type=int, default=42,
                        help="Random seed for significance tests")

    # Ablation-specific (we will write results into model_dir as well)
    parser.add_argument("--ablation_epochs", type=int, default=20,
                        help="Epochs for GNNs in ablation study")
    parser.add_argument("--ablation_batch_size", type=int, default=256,
                        help="Batch size for ablation GNNs")
    parser.add_argument("--ablation_num_neighbors", type=int, nargs="+", default=[15, 15, 10],
                        help="Neighbours per layer for ablation GNNs")
    
    # Sensitivity-specific
    parser.add_argument(
        "--sens_param", type=str, default="hidden_dim",
        choices=["hidden_dim", "num_layers", "heads", "num_neighbors", "batch_size", "epochs"],
        help="Hyperparameter to vary in sensitivity analysis",
    )
    parser.add_argument(
        "--sens_values", type=int, nargs="+", default=[64, 128, 256, 512],
        help="Values to evaluate for the chosen sensitivity hyperparameter",
    )


    # Stage toggles
        # Stage toggles
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip main training (train.py)")
    parser.add_argument("--skip_hybrid", action="store_true",
                        help="Skip hybrid training (train_hybrid.py)")
    parser.add_argument("--skip_sensitivity", action="store_true",
                        help="Skip sensitivity analysis")
    parser.add_argument("--skip_shap", action="store_true",
                        help="Skip SHAP analysis")
    parser.add_argument("--skip_ablation", action="store_true",
                        help="Skip ablation study")
    parser.add_argument("--skip_significance", action="store_true",
                        help="Skip significance tests")
    parser.add_argument("--skip_aggregate", action="store_true",
                        help="Skip aggregation of analysis outputs")


    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    python_exe = sys.executable

    # 1. Train baseline + GAT
    if not args.skip_train:
        cmd = [
            python_exe, "train.py",
            "--root", args.root,
            "--model_dir", args.model_dir,
            "--agg_method", args.agg_method,
            "--hidden_dim", str(args.hidden_dim),
            "--num_layers", str(args.num_layers),
            "--heads", str(args.heads),
            "--batch_size", str(args.batch_size),
            "--epochs", str(args.epochs),
        ]
        # num_neighbors is a list
        cmd.extend(["--num_neighbors"] + [str(n) for n in args.num_neighbors])
        if args.no_add_degree:
            cmd.append("--no_add_degree")

        run_cmd(cmd, "Main training (baseline + GAT)")

        # 2. Train hybrid models on top of GAT embeddings
    if not args.skip_hybrid:
        cmd = [
            python_exe, "train_hybrid.py",
            "--root", args.root,
            "--model_dir", args.model_dir,
            "--agg_method", args.agg_method,
            "--hidden_dim", str(args.hidden_dim),
            "--num_layers", str(args.num_layers),
            "--heads", str(args.heads),
            "--classifier", "all",
        ]
        if args.no_add_degree:
            cmd.append("--no_add_degree")

        run_cmd(cmd, "Hybrid training (GAT + classical models)")

    # 3. Sensitivity analysis (GAT and hybrid)
    if not args.skip_sensitivity:
        cmd = [
            python_exe, "sensitivity_analysis.py",
            "--root", args.root,
            "--model_dir", args.model_dir,
            "--agg_method", args.agg_method,
            "--param", args.sens_param,
            "--hidden_dim", str(args.hidden_dim),
            "--num_layers", str(args.num_layers),
            "--heads", str(args.heads),
            "--batch_size", str(args.batch_size),
            "--epochs", str(args.epochs),
        ]
        # num_neighbors is a list
        cmd.extend(["--num_neighbors"] + [str(n) for n in args.num_neighbors])
        # sensitivity values to sweep
        cmd.extend(["--values"] + [str(v) for v in args.sens_values])
        if args.no_add_degree:
            cmd.append("--no_add_degree")

        run_cmd(cmd, f"Sensitivity analysis for {args.sens_param}")

    # 4. SHAP analysis for chosen hybrid model
    if not args.skip_shap:
        cmd = [
            python_exe, "shap_analysis.py",
            "--root", args.root,
            "--model_dir", args.model_dir,
            "--agg_method", args.agg_method,
            "--hidden_dim", str(args.hidden_dim),
            "--num_layers", str(args.num_layers),
            "--model_tag", args.shap_model_tag,
            "--split", args.shap_split,
            "--nsamples", str(args.shap_nsamples),
            "--max_labels", str(args.shap_max_labels),
        ]
        if args.no_add_degree:
            cmd.append("--no_add_degree")

        run_cmd(cmd, f"SHAP analysis for {args.shap_model_tag}")


    # 5. Ablation study (results also stored under model_dir)
    if not args.skip_ablation:
        cmd = [
            python_exe, "ablation_study.py",
            "--root", args.root,
            "--out_dir", args.model_dir,  # keep everything in one place
            "--epochs", str(args.ablation_epochs),
            "--batch_size", str(args.ablation_batch_size),
        ]
        cmd.extend(["--num_neighbors"] + [str(n) for n in args.ablation_num_neighbors])

        run_cmd(cmd, "Ablation study")

    # 6. Statistical significance tests
    if not args.skip_significance:
        cmd = [
            python_exe, "significance_tests.py",
            "--root", args.root,
            "--model_dir", args.model_dir,
            "--split", args.sig_split,
            "--n_bootstrap", str(args.sig_n_bootstrap),
            "--seed", str(args.sig_seed),
        ]
        run_cmd(cmd, f"Significance tests on {args.sig_split} split")

    # 7. Aggregate all available analysis artefacts
    if not args.skip_aggregate:
        cmd = [
            python_exe, "aggregate_analysis.py",
            "--root_dir", args.model_dir,
        ]
        run_cmd(cmd, "Aggregating SHAP, ablation, and significance outputs")

    print("\n[ORCH] Pipeline complete.")


if __name__ == "__main__":
    main()
