#!/usr/bin/env python
"""
plot_analysis_diagrams.py

Generate all key diagrams (SHAP, ablation, significance tests, and
hyperparameter sensitivity) from the outputs produced by:

  - shap_analysis.py
  - ablation_study.py
  - significance_tests.py
  - sensitivity_analysis.py
  - aggregate_analysis.py   (for file locations / structure only)

This script assumes you have already run the numeric pipeline, e.g.:

  python orchestrate_pipeline.py \
      --root ./data \
      --model_dir ./models_main \
      --agg_method mean \
      --hidden_dim 256 \
      --num_layers 3 \
      --batch_size 2048 \
      --num_neighbors 25 15 10 \
      --epochs 30

and (optionally) a sensitivity run such as:

  python sensitivity_analysis.py \
      --root ./data \
      --model_dir ./models_main \
      --param hidden_dim --values 64 128 256 512

Outputs are written under:

  ROOT_DIR/figures/
      shap/
      ablation/
      significance/
      sensitivity/
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# SHAP plots
# ---------------------------------------------------------------------


def plot_shap_global_importance(
    root_dir: str,
    model_tag: str,
    top_k: int,
    out_dir: str,
) -> Optional[str]:
    shap_dir = os.path.join(root_dir, "shap")
    global_path = os.path.join(shap_dir, f"{model_tag}_global_importance.npy")
    summary_path = os.path.join(shap_dir, f"{model_tag}_shap_summary.json")

    if not os.path.exists(global_path):
        print(f"[SHAP] Global importance file not found: {global_path}")
        return None

    global_imp = np.load(global_path)
    d = global_imp.shape[0]
    k = min(top_k, d)

    indices = np.argsort(global_imp)[::-1][:k]
    values = global_imp[indices]

    split = "?"
    summary = load_json(summary_path)
    if summary is not None:
        split = summary.get("split", "?")

    ensure_dir(out_dir)
    fig_path = os.path.join(out_dir, f"{model_tag}_global_top{k}.png")

    plt.figure(figsize=(max(6, 0.4 * k), 4))
    plt.bar(np.arange(k), values)
    plt.xticks(np.arange(k), [str(i) for i in indices], rotation=45, ha="right")
    plt.ylabel("Mean |SHAP value|")
    plt.xlabel("Feature index")
    plt.title(f"Top-{k} global SHAP importances ({model_tag}, split={split})")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"[SHAP] Global importance figure saved to {fig_path}")
    return fig_path


def plot_shap_per_label_heatmap(
    root_dir: str,
    model_tag: str,
    top_k_features: int,
    max_labels: int,
    out_dir: str,
) -> Optional[str]:
    shap_dir = os.path.join(root_dir, "shap")
    global_path = os.path.join(shap_dir, f"{model_tag}_global_importance.npy")
    per_label_path = os.path.join(shap_dir, f"{model_tag}_per_label_importance.npy")

    if not (os.path.exists(global_path) and os.path.exists(per_label_path)):
        print(f"[SHAP] Per-label SHAP files not found for tag {model_tag}.")
        return None

    global_imp = np.load(global_path)
    per_label = np.load(per_label_path)  # [n_labels, d]

    n_labels, d = per_label.shape
    k_feat = min(top_k_features, d)
    k_labels = min(max_labels, n_labels)

    top_feat_indices = np.argsort(global_imp)[::-1][:k_feat]
    submatrix = per_label[:k_labels, :][:, top_feat_indices]

    ensure_dir(out_dir)
    fig_path = os.path.join(out_dir, f"{model_tag}_heatmap_labels{k_labels}_feat{k_feat}.png")

    plt.figure(figsize=(max(6, 0.4 * k_feat), max(4, 0.2 * k_labels)))
    im = plt.imshow(submatrix, aspect="auto", interpolation="nearest")
    plt.colorbar(im, label="Mean |SHAP value|")

    plt.xlabel("Feature index (top global)")
    plt.ylabel("Label index")
    plt.xticks(
        np.arange(k_feat),
        [str(i) for i in top_feat_indices],
        rotation=45,
        ha="right",
    )
    plt.yticks(np.arange(k_labels), [str(i) for i in range(k_labels)])
    plt.title(f"Per-label SHAP importances ({model_tag})")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"[SHAP] Per-label heatmap figure saved to {fig_path}")
    return fig_path


def generate_shap_figures(
    root_dir: str,
    model_tag: str,
    top_k: int,
    max_labels: int,
    figures_root: str,
) -> None:
    shap_fig_dir = os.path.join(figures_root, "shap")
    ensure_dir(shap_fig_dir)

    print(f"[SHAP] Generating SHAP plots for tag={model_tag}")
    plot_shap_global_importance(root_dir, model_tag, top_k, shap_fig_dir)
    plot_shap_per_label_heatmap(root_dir, model_tag, top_k, max_labels, shap_fig_dir)


# ---------------------------------------------------------------------
# Ablation plots
# ---------------------------------------------------------------------


def generate_ablation_figures(root_dir: str, figures_root: str) -> None:
    ablation_path = os.path.join(root_dir, "ablation_results.json")
    if not os.path.exists(ablation_path):
        print(f"[ABLATION] No ablation_results.json found at {ablation_path}. Skipping.")
        return

    with open(ablation_path, "r") as f:
        results = json.load(f)

    baseline_cfgs = [cfg for cfg in results if cfg.get("model_family") == "baseline"]
    gnn_cfgs = [cfg for cfg in results if cfg.get("model_family") == "gnn"]

    if not baseline_cfgs and not gnn_cfgs:
        print("[ABLATION] No recognised configurations in ablation_results.json.")
        return

    ablation_fig_dir = os.path.join(figures_root, "ablation")
    ensure_dir(ablation_fig_dir)

    if baseline_cfgs:
        names = [cfg.get("config_name", "?") for cfg in baseline_cfgs]
        val_auc = [cfg.get("val_auc", np.nan) for cfg in baseline_cfgs]
        test_auc = [cfg.get("test_auc", np.nan) for cfg in baseline_cfgs]

        x = np.arange(len(names))
        width = 0.35

        fig_path = os.path.join(ablation_fig_dir, "baseline_val_test_auc.png")
        plt.figure(figsize=(max(6, 0.6 * len(names)), 4))
        plt.bar(x - width / 2, val_auc, width, label="Val AUC")
        plt.bar(x + width / 2, test_auc, width, label="Test AUC")

        plt.xticks(x, names, rotation=45, ha="right")
        plt.ylabel("Macro ROC-AUC")
        plt.title("Baseline ablation: ROC-AUC per configuration")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()

        print(f"[ABLATION] Baseline AUC figure saved to {fig_path}")

    if gnn_cfgs:
        names = [cfg.get("config_name", "?") for cfg in gnn_cfgs]
        gnn_test_auc = [cfg.get("gnn_test_auc", np.nan) for cfg in gnn_cfgs]
        hybrid_test_auc = [cfg.get("hybrid_xgb_test_auc", np.nan) for cfg in gnn_cfgs]

        x = np.arange(len(names))
        width = 0.35

        fig_path = os.path.join(ablation_fig_dir, "gnn_vs_hybrid_test_auc.png")
        plt.figure(figsize=(max(6, 0.6 * len(names)), 4))
        plt.bar(x - width / 2, gnn_test_auc, width, label="GNN test AUC")
        plt.bar(x + width / 2, hybrid_test_auc, width, label="Hybrid XGB test AUC")

        plt.xticks(x, names, rotation=45, ha="right")
        plt.ylabel("Macro ROC-AUC")
        plt.title("GNN vs Hybrid-XGB: test ROC-AUC per configuration")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()

        print(f"[ABLATION] GNN vs Hybrid AUC figure saved to {fig_path}")


# ---------------------------------------------------------------------
# Significance plots
# ---------------------------------------------------------------------


def generate_significance_figures(root_dir: str, figures_root: str) -> None:
    sig_paths = glob.glob(os.path.join(root_dir, "significance_results_*.json"))
    if not sig_paths:
        print("[SIG] No significance_results_*.json files found. Skipping.")
        return

    sig_fig_dir = os.path.join(figures_root, "significance")
    ensure_dir(sig_fig_dir)

    for path in sig_paths:
        split = os.path.splitext(os.path.basename(path))[0].replace(
            "significance_results_", ""
        )
        with open(path, "r") as f:
            res_list = json.load(f)

        if not res_list:
            print(f"[SIG] Empty results in {path}, skipping.")
            continue

        labels = []
        diffs = []
        lower_err = []
        upper_err = []

        for res in res_list:
            a = res.get("model_a", "?")
            b = res.get("model_b", "?")
            label = f"{a} vs {b}"
            diff = res.get("mean_diff_auc", np.nan)
            ci_low = res.get("ci_low", np.nan)
            ci_high = res.get("ci_high", np.nan)

            labels.append(label)
            diffs.append(diff)
            lower_err.append(diff - ci_low)
            upper_err.append(ci_high - diff)

        y_pos = np.arange(len(labels))
        yerr = [lower_err, upper_err]

        fig_path = os.path.join(sig_fig_dir, f"auc_diff_ci_{split}.png")

        plt.figure(figsize=(8, max(3, 0.5 * len(labels))))
        plt.errorbar(
            diffs,
            y_pos,
            xerr=yerr,
            fmt="o",
            capsize=4,
        )
        plt.axvline(0.0, linestyle="--", linewidth=1)
        plt.yticks(y_pos, labels)
        plt.xlabel("Mean AUC difference (model_b - model_a)")
        plt.title(f"AUC differences with bootstrap CIs (split={split})")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()

        print(f"[SIG] Significance figure for split={split} saved to {fig_path}")


# ---------------------------------------------------------------------
# Sensitivity plots
# ---------------------------------------------------------------------


def generate_sensitivity_figures(root_dir: str, figures_root: str) -> None:
    """
    Generate hyperparameter sensitivity diagrams from `sensitivity.json`
    produced by collect_sensitivity_from_saved.py.

    Expects a list of dicts, each with at least:
      - param            (e.g. "hidden_dim")
      - param_value      (e.g. 32, 64, 128, 256)
      - gnn_val_auc
      - gnn_test_auc
      - hybrid_xgb_val_auc  (may be None)
      - hybrid_xgb_test_auc (may be None)
    """
    sens_path = os.path.join(root_dir, "sensitivity.json")
    if not os.path.exists(sens_path):
        print(f"[SENS] No sensitivity.json found at {sens_path}. Skipping.")
        return

    with open(sens_path, "r") as f:
        all_results = json.load(f)

    if not all_results:
        print("[SENS] sensitivity.json is empty. Skipping.")
        return

    sens_fig_dir = os.path.join(figures_root, "sensitivity")
    os.makedirs(sens_fig_dir, exist_ok=True)

    # Group entries by param (e.g. hidden_dim, num_layers, ...)
    by_param = {}
    for entry in all_results:
        p = entry.get("param", "unknown")
        by_param.setdefault(p, []).append(entry)

    for param, entries in by_param.items():
        # Sort entries by param_value
        vals = [e["param_value"] for e in entries]

        # Convert to float for sorting; fall back to original if needed
        try:
            order = np.argsort(np.array(vals, dtype=float))
        except ValueError:
            order = np.argsort(np.array([str(v) for v in vals]))

        vals_sorted = [vals[i] for i in order]

        gnn_val = [entries[i].get("gnn_val_auc", np.nan) for i in order]
        gnn_test = [entries[i].get("gnn_test_auc", np.nan) for i in order]
        hyb_val = [entries[i].get("hybrid_xgb_val_auc", np.nan) for i in order]
        hyb_test = [entries[i].get("hybrid_xgb_test_auc", np.nan) for i in order]

        has_hybrid = not all(v is None or np.isnan(v) for v in hyb_val + hyb_test)

        fig_path = os.path.join(sens_fig_dir, f"sensitivity_{param}.png")

        plt.figure(figsize=(6, 4))
        # GNN curves
        plt.plot(vals_sorted, gnn_val, marker="o", label="GNN val AUC")
        plt.plot(vals_sorted, gnn_test, marker="o", linestyle="--", label="GNN test AUC")

        # Hybrid curves (if available)
        if has_hybrid:
            plt.plot(vals_sorted, hyb_val, marker="s", label="Hybrid-XGB val AUC")
            plt.plot(vals_sorted, hyb_test, marker="s", linestyle="--",
                     label="Hybrid-XGB test AUC")

        plt.xlabel(param)
        plt.ylabel("Macro ROC-AUC")
        plt.title(f"Hyperparameter sensitivity: {param}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()

        print(f"[SENS] Sensitivity figure for param={param} saved to {fig_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate SHAP, ablation, significance, and sensitivity diagrams "
            "from existing BioGraphFusion / ogbn-proteins analysis artefacts."
        )
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help=(
            "Directory containing analysis outputs: "
            "ablation_results.json, significance_results_*.json, shap/, etc. "
            "Typically this is the same as --model_dir in orchestrate_pipeline.py."
        ),
    )
    parser.add_argument(
        "--shap_model_tag",
        type=str,
        default="graphsage_xgb",
        help=(
            "Which hybrid model tag to use for SHAP plots "
            "(must match the tag used in shap_analysis.py, "
            "e.g. 'graphsage_xgb' or 'graphsage_rf')."
        ),
    )
    parser.add_argument(
        "--shap_top_k",
        type=int,
        default=20,
        help="Number of top global features to plot in SHAP diagrams.",
    )
    parser.add_argument(
        "--shap_max_labels",
        type=int,
        default=20,
        help="Maximum number of labels to show in the SHAP heatmap.",
    )

    args = parser.parse_args()

    root_dir = args.root_dir
    figures_root = os.path.join(root_dir, "figures")
    ensure_dir(figures_root)

    print(f"[MAIN] Root analysis dir: {root_dir}")
    print(f"[MAIN] Figures will be written under: {figures_root}")

    generate_shap_figures(
        root_dir=root_dir,
        model_tag=args.shap_model_tag,
        top_k=args.shap_top_k,
        max_labels=args.shap_max_labels,
        figures_root=figures_root,
    )

    generate_ablation_figures(root_dir=root_dir, figures_root=figures_root)
    generate_significance_figures(root_dir=root_dir, figures_root=figures_root)
    generate_sensitivity_figures(root_dir=root_dir, figures_root=figures_root)

    print("[MAIN] Diagram generation complete.")


if __name__ == "__main__":
    main()
