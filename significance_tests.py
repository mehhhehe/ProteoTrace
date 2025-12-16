# significance_tests.py

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score

try:
    from scipy.stats import ttest_rel, wilcoxon
except ImportError:
    ttest_rel = wilcoxon = None


from data_loader import load_raw_ogbn_proteins


def load_probs(model_dir: str, tag: str, split: str) -> np.ndarray:
    path = os.path.join(model_dir, f"{tag}_probs_{split}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Probability file {path} not found.")
    return np.load(path)


def per_label_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> np.ndarray:
    n_labels = y_true.shape[1]
    aucs = np.zeros(n_labels, dtype=np.float64)
    for i in range(n_labels):
        try:
            aucs[i] = roc_auc_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            # all-positive or all-negative -> undefined AUC; mark as NaN
            aucs[i] = np.nan
    return aucs


def bootstrap_ci(
    diffs: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
) -> Dict[str, float]:
    diffs = diffs[~np.isnan(diffs)]
    n = diffs.shape[0]
    if n == 0:
        return {"mean_diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    boot_means = np.zeros(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        boot_means[b] = diffs[idx].mean()
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(lower),
        "ci_high": float(upper),
    }


def compare_models(
    y_true: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    name_a: str,
    name_b: str,
) -> Dict:
    auc_a = per_label_auc(y_true, probs_a)
    auc_b = per_label_auc(y_true, probs_b)
    diffs = auc_b - auc_a

    stats = bootstrap_ci(diffs)

    if ttest_rel is not None:
        valid = ~np.isnan(diffs)
        t_stat, t_p = ttest_rel(auc_a[valid], auc_b[valid])
    else:
        t_stat = float("nan")
        t_p = float("nan")

    if wilcoxon is not None:
        valid = ~np.isnan(diffs)
        try:
            w_stat, w_p = wilcoxon(auc_a[valid], auc_b[valid])
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
    else:
        w_stat = float("nan")
        w_p = float("nan")

    return {
        "model_a": name_a,
        "model_b": name_b,
        "mean_auc_a": float(np.nanmean(auc_a)),
        "mean_auc_b": float(np.nanmean(auc_b)),
        "mean_diff_auc": stats["mean_diff"],
        "ci_low": stats["ci_low"],
        "ci_high": stats["ci_high"],
        "ttest_stat": float(t_stat),
        "ttest_p": float(t_p),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p": float(w_p),
        "n_labels_used": int((~np.isnan(diffs)).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistical significance tests between models")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["valid", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_bootstrap", type=int, default=5000)
    args = parser.parse_args()

    np.random.seed(args.seed)

    graph, labels, split_idx = load_raw_ogbn_proteins(args.root)
    if args.split == "test":
        idx = split_idx["test"]
    else:
        idx = split_idx["valid"]
    y_true = labels[idx]

    # Load probability matrices
    baseline_probs = load_probs(args.model_dir, "baseline", args.split)
    gnn_probs = load_probs(args.model_dir, "gat", args.split)

    results = []

    # Baseline vs GNN
    print("[Significance] Baseline vs GAT")
    res_bg = compare_models(
        y_true,
        baseline_probs,
        gnn_probs,
        name_a="baseline",
        name_b="gat",
    )
    results.append(res_bg)

    # Hybrid XGB if available
    xgb_path = os.path.join(args.model_dir, f"gat_xgb_probs_{args.split}.npy")
    if os.path.exists(xgb_path):
        hybrid_probs = np.load(xgb_path)
        print("[Significance] Baseline vs Hybrid XGB")
        res_bh = compare_models(
            y_true,
            baseline_probs,
            hybrid_probs,
            name_a="baseline",
            name_b="gat_xgb",
        )
        results.append(res_bh)

        print("[Significance] GAT vs Hybrid XGB")
        res_gh = compare_models(
            y_true,
            gnn_probs,
            hybrid_probs,
            name_a="gat",
            name_b="gat_xgb",
        )
        results.append(res_gh)
    else:
        print("[Significance] Hybrid XGB probabilities not found; skipping hybrid comparisons.")

    out_path = os.path.join(args.model_dir, f"significance_results_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Significance results saved to {out_path}")


if __name__ == "__main__":
    main()
