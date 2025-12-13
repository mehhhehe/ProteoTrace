# aggregate_analysis.py

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List


def safe_load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate SHAP, ablation, and significance analyses")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Directory containing ablation_results.json, significance_results_*.json, shap/ etc.")
    args = parser.parse_args()

    root = args.root_dir

    # 1) Ablation results
    ablation_path = os.path.join(root, "ablation_results.json")
    ablation = safe_load_json(ablation_path)

    # 2) Significance results
    signif_paths = glob.glob(os.path.join(root, "significance_results_*.json"))
    significance = {}
    for p in signif_paths:
        split = os.path.splitext(os.path.basename(p))[0].replace("significance_results_", "")
        significance[split] = safe_load_json(p)

    # 3) SHAP summaries
    shap_dir = os.path.join(root, "shap")
    shap_summaries = {}
    if os.path.isdir(shap_dir):
        for p in glob.glob(os.path.join(shap_dir, "*_shap_summary.json")):
            tag = os.path.splitext(os.path.basename(p))[0].replace("_shap_summary", "")
            shap_summaries[tag] = safe_load_json(p)

    summary = {
        "ablation_results": ablation,
        "significance_results": significance,
        "shap_summaries": shap_summaries,
    }

    # Save combined JSON
    out_json = os.path.join(root, "analysis_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Also emit a simple Markdown report
    out_md = os.path.join(root, "analysis_report.md")
    with open(out_md, "w") as f:
        f.write("# Analysis Summary\n\n")

        if ablation is not None:
            f.write("## Ablation Study\n\n")
            for cfg in ablation:
                f.write(f"- **{cfg.get('config_name','?')}**: {cfg}\n")
            f.write("\n")

        if significance:
            f.write("## Significance Tests\n\n")
            for split, res_list in significance.items():
                f.write(f"### Split: {split}\n\n")
                if res_list is None:
                    continue
                for res in res_list:
                    f.write(
                        f"- {res['model_a']} vs {res['model_b']}: "
                        f"mean_diff_auc={res['mean_diff_auc']:.4f}, "
                        f"CI[{res['ci_low']:.4f}, {res['ci_high']:.4f}], "
                        f"ttest_p={res['ttest_p']:.3e}, "
                        f"wilcoxon_p={res['wilcoxon_p']:.3e}\n"
                    )
                f.write("\n")

        if shap_summaries:
            f.write("## SHAP Summaries\n\n")
            for tag, s in shap_summaries.items():
                if s is None:
                    continue
                f.write(f"- **{tag}** (split={s.get('split','?')}): "
                        f"top10_dims={s.get('global_importance_top10_indices', [])}\n")
            f.write("\n")

    print(f"Combined analysis written to {out_json} and {out_md}")


if __name__ == "__main__":
    main()
