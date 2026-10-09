#!/usr/bin/env python
"""m13 -- the remaining named plots required by the spec.

Each title carries model, dataset, condition and n, so a figure cannot be lifted out of
the report and re-captioned without the caption becoming visibly wrong.

Usage: python scripts/m13_plots.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v2 as V      # noqa: E402

ROOT = V.ROOT
PLOTS = ROOT / "PLOTS"
R = ROOT / "reports"


def _clean_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS.mkdir(exist_ok=True)
    made = []

    # ---- 07 behaviour, clean vs no_image, both checkpoints, n=900 ---------------- #
    d = _clean_json(R / ".." / "RESULTS" / "README.json") if False else None
    import csv
    beh = list(csv.DictReader((ROOT / "RESULTS" / "behaviour_per_example.csv")
                              .open(encoding="utf-8")))
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    models = ["qwen2vl", "medvlmr1"]
    conds = ["clean", "no_image"]
    w = 0.34
    for i, c in enumerate(conds):
        vals, errs = [], []
        for m in models:
            sel = [r for r in beh if r["model"] == m and r["condition"] == c
                   and r["order"] == "O0"]
            acc = np.mean([float(r["correct"]) for r in sel]) if sel else np.nan
            vals.append(100 * acc)
            errs.append(0.0)
        ax.bar(np.arange(len(models)) + (i - 0.5) * w, vals, w,
               label=f"{c} (n={len([r for r in beh if r['condition']==c and r['order']=='O0'])//len(models)})",
               color=("tab:green" if c == "clean" else "tab:red"))
        for x, v in zip(np.arange(len(models)) + (i - 0.5) * w, vals):
            ax.text(x, v + 1, f"{v:.1f}", ha="center", fontsize=9)
    ax.axhline(25, color="k", linestyle="--", linewidth=1.2)
    ax.text(1.42, 26, "4-option chance 25%", fontsize=8, ha="right")
    ax.axhline(50, color="k", linestyle=":", linewidth=1.2)
    ax.text(1.42, 51, "2-option chance 50%", fontsize=8, ha="right")
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels([V.MODEL_DISPLAY[m] for m in models], fontsize=9)
    ax.set_ylabel("accuracy, order O0 (%)")
    ax.set_title("Behaviour: clean vs no_image\n"
                 "dataset=current900 · models=Qwen2-VL-2B, MedVLM-R1 · "
                 "condition=clean|no_image · n=900 each (825 four-option, 75 two-option)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    f = PLOTS / "07_behaviour_clean_vs_no_image__current900__n900.png"
    fig.savefig(f, dpi=140)
    plt.close(fig)
    made.append(f.name)

    # ---- 08 external VQA-RAD by condition, n=251 --------------------------------- #
    runs = []
    for p in sorted(glob.glob(str(ROOT / "external" / "x1_vqarad_yesno_*.json"))):
        dd = _clean_json(Path(p))
        if dd.get("n_items", 0) >= 100:
            runs.append(dd)
    order = ["qwen2vl", "medvlmr1", "qwen25vl3b", "mvt_sft", "mvt_rl"]
    runs.sort(key=lambda x: order.index(x["model"]) if x["model"] in order else 99)
    if runs:
        fig, ax = plt.subplots(figsize=(9.6, 4.8))
        conds = ["clean", "shuffled_image", "no_image"]
        colors = {"clean": "tab:green", "shuffled_image": "tab:orange",
                  "no_image": "tab:red"}
        w = 0.26
        for i, c in enumerate(conds):
            vals = [100 * r["accuracy"][c]["mean_per_order"] for r in runs]
            ax.bar(np.arange(len(runs)) + (i - 1) * w, vals, w, label=c,
                   color=colors[c])
            for x, v in zip(np.arange(len(runs)) + (i - 1) * w, vals):
                ax.text(x, v + 0.6, f"{v:.1f}", ha="center", fontsize=7.5)
        ax.axhline(50, color="k", linestyle="--", linewidth=1.2)
        ax.text(len(runs) - 0.5, 50.8, "chance 50%", fontsize=8, ha="right")
        ax.axhline(100 * runs[0]["label_prior_baseline"], color="k",
                   linestyle=":", linewidth=1.2)
        ax.text(len(runs) - 0.5, 100 * runs[0]["label_prior_baseline"] + 0.8,
                f"majority prior {100*runs[0]['label_prior_baseline']:.1f}%",
                fontsize=8, ha="right")
        ax.set_xticks(np.arange(len(runs)))
        ax.set_xticklabels([r["model"] for r in runs], fontsize=9)
        ax.set_ylabel("accuracy (%)  mean-per-order")
        ax.set_title("External: VQA-RAD yes/no\n"
                     "dataset=VQA-RAD · 5 released checkpoints · "
                     "conditions=clean|shuffled_image|no_image · n=251 items / 135 images")
        ax.legend(fontsize=8)
        fig.tight_layout()
        f = PLOTS / ("08_external_vqarad_by_condition__vqarad__5checkpoints__n251.png")
        fig.savefig(f, dpi=140)
        plt.close(fig)
        made.append(f.name)

    # ---- 09/10 corrected mismatch probes and their controls ---------------------- #
    p0 = ROOT / "probes" / "P0_FIXED_PROBES.json"
    if p0.is_file():
        d0 = _clean_json(p0)
        # Refuse to draw the corrected figures from a stale file. The first version of
        # this script did exactly that and produced a control plot showing 0.057 instead
        # of 0.5, i.e. the figure silently disagreed with the report beside it.
        src = set()
        for m in d0.get("models", {}).values():
            if isinstance(m, dict) and "vision_only_negative_control" in m:
                for v in m["vision_only_negative_control"].values():
                    if isinstance(v, dict):
                        src.add(v.get("folds_source") or "(absent)")
        # `folds_source` absent means the file predates the BUG-9 fix, which is just as
        # stale as a file that names the unaligned fallback. Only the aligned provenance
        # may be published.
        if not src or not all(s == "derangement manifest (aligned)" for s in src):
            print("!! P0_FIXED_PROBES.json is not the aligned-fold run "
                  f"(provenance seen: {sorted(src) or 'none'}). Regenerating plots 09 "
                  "and 10 would publish superseded numbers, so they are skipped.")
            return 1
        SHORT = {"C_mismatch_balanced": "balanced mismatch\n(must be 0.500)",
                 "C_mismatch_old_parent_pool_donor": "old parent-pool\ndonor",
                 "A_blur": "blur",
                 "D_to_same_answer_donor": "→ same-answer\ndonor",
                 "D_to_opposite_answer_donor": "→ opposite-answer\ndonor"}
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6))
        for ax, m in zip(axes, [k for k in V.MODELS if k in d0["models"]]):
            bm = d0["models"][m]
            if "error" in bm:
                continue
            T = bm["T"]
            lay = list(range(T))
            def curve(key):
                return [bm["by_layer"][str(t)][key]["auroc"]
                        if key in bm["by_layer"][str(t)] else np.nan for t in lay]
            ax.plot(lay, curve("B_frozen_transfer_to_modality_specific"), marker="o",
                    ms=3, label="B frozen transfer → modality-specific")
            ax.plot(lay, curve("B_frozen_transfer_permuted_control"), marker="s",
                    ms=3, linestyle="--", label="B permuted control")
            ax.plot(lay, curve("C_mismatch_balanced_decision"), marker="^", ms=3,
                    label="C balanced mismatch")
            ax.plot(lay, curve("C_mismatch_balanced_shuffled_control"), marker="v",
                    ms=3, linestyle="--", label="C shuffled control")
            ax.axhline(0.5, color="k", linewidth=1)
            ax.set_ylim(0.4, 1.02)
            ax.set_xlabel("stored hidden-state row  (row 0 = input embedding)")
            ax.set_title(f"{V.MODEL_DISPLAY[m]}")
            ax.legend(fontsize=7)
        axes[0].set_ylabel("AUROC, 5-fold source-grouped CV")
        fig.suptitle("Corrected probes: frozen transfer and balanced mismatch\n"
                     "dataset=current900 · models=Qwen2-VL-2B, MedVLM-R1 · "
                     "conditions=clean|blur_severe|modality_specific_severe|"
                     "balanced_mismatch · n=900 each")
        fig.tight_layout()
        f = PLOTS / "09_probe_curves_corrected__current900__all_rows__n900.png"
        fig.savefig(f, dpi=140)
        plt.close(fig)
        made.append(f.name)

        fig, ax = plt.subplots(figsize=(9.2, 4.4))
        keys, vals, cols = [], [], []
        for m in [k for k in V.MODELS if k in d0["models"]]:
            bm = d0["models"][m]
            if "error" in bm:
                continue
            for k, v in bm["vision_only_negative_control"].items():
                if not isinstance(v, dict) or v.get("auroc") is None:
                    continue
                keys.append(f"{m}\n{SHORT.get(k, k)}")
                vals.append(v["auroc"])
                cols.append("tab:green" if k == "C_mismatch_balanced" else "tab:red")
        x = np.arange(len(keys))
        ax.bar(x, vals, 0.6, color=cols)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
        ax.axhline(0.5, color="k", linestyle="--", linewidth=1.4)
        ax.text(len(keys) - 0.5, 0.52, "required 0.5 for a balanced pair", fontsize=8,
                ha="right")
        ax.set_xticks(x)
        ax.set_xticklabels(keys, fontsize=7.5)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("AUROC from the vision-encoder vector alone")
        ax.set_title("Vision-only negative controls (representation carries no question)\n"
                     "dataset=current900 · models=Qwen2-VL-2B, MedVLM-R1 · "
                     "folds aligned with the derangement · green = balanced pair "
                     "(required 0.500, n=900) · red = donor families (n=529 / 73, "
                     "not balanced)")
        fig.tight_layout()
        f = PLOTS / "10_vision_only_controls__current900__n900.png"
        fig.savefig(f, dpi=140)
        plt.close(fig)
        made.append(f.name)

    print(f"wrote {len(made)} plots to PLOTS/")
    for m in made:
        print(f"  {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
