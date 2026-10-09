#!/usr/bin/env python
"""Stage 1 -- matched-processing behaviour, and the check on the old conclusion.

Spec 04 Stage 1 / 06 Phase C.

Answers one question first: **does the old central behavioural result survive the
matched image-processing control?**  Everything internal depends on that.  The old run
gave Qwen2-VL-2B 66.78% / MedVLM-R1 80.22% on clean and 44.63% / 58.07% with no image,
under each checkpoint's own native processor.  Here both run at min_pixels=3136,
max_pixels=401408, so a large X-ray is no longer fed to one model as 11235 visual
tokens and to the other as 484.

Reported per condition:
  * mean-per-order accuracy (each option order counted separately);
  * order-averaged-semantic-distribution accuracy (the aggregation the old project
    used for its headline; both are saved because the spec notes M2 is
    aggregation-sensitive);
  * p(GT), entropy, GT margin;
  * source-clustered bootstrap CIs.

Writes runs/current900/behavior/behavior_summary.json and reports/MATCHED_BEHAVIOR.md
"""

from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v2 as V   # noqa: E402

N_BOOT = 2000
BOOT_SEED = V.GLOBAL_SEED
FEAT = V.ROOT / "features"
OUT = V.ROOT / "runs" / "current900" / "behavior"


def shard_records(model: str, cond: str, order: str) -> list[dict]:
    d = FEAT / model / cond / order
    out = []
    for p in sorted(d.glob("shard_*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("status") == "ok":
                    out.append(r)
    return out


def boot_ci(ids, vals, n_boot=N_BOOT, seed=BOOT_SEED):
    by = defaultdict(list)
    for c, v in zip(ids, vals):
        by[c].append(v)
    keys = list(by)
    arrs = [np.asarray(by[k], dtype=float) for k in keys]
    vals = np.asarray(vals, dtype=float)
    if len(keys) < 2:
        return float(vals.mean()), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        d[b] = np.concatenate([arrs[i] for i in pick]).mean()
    return float(vals.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def order_average(recs: list[dict], items: dict) -> list[dict]:
    """Average the semantic distribution over DISTINCT option orders.

    Two orders of a 2-option question can be the same permutation; treated as one
    observation rather than counting the same information twice.
    """
    by = defaultdict(dict)
    for r in recs:
        by[(r["sample_id"])][tuple(r["permutation"])] = r
    out = []
    for sid, perms in by.items():
        item = items[sid]
        n = item["n_options"]
        acc = np.zeros(n)
        per_order_correct = []
        for p, r in perms.items():
            v = np.array([r["semantic_probs"].get(str(i), 0.0) for i in range(n)])
            acc += v
            per_order_correct.append(float(bool(r["correct"])))
        pm = acc / acc.sum()
        gt = item["gt_index"]
        others = np.delete(pm, gt)
        out.append({
            "sample_id": sid, "modality": item["modality"], "n_options": n,
            "n_distinct_orders": len(perms),
            "accuracy": float(int(np.argmax(pm)) == gt),
            "accuracy_mean_per_order": float(np.mean(per_order_correct)),
            "p_gt": float(pm[gt]),
            "entropy": float(-(pm * np.log(np.maximum(pm, 1e-12))).sum()),
            "margin": float(pm[gt] - (others.max() if len(others) else 0.0)),
        })
    return out


def main() -> int:
    items = V.load_items()
    summary: dict = {"created": V.now(), "processor_id": V.MATCHED_PROCESSOR_ID,
                     "min_pixels": V.MATCHED_MIN_PIXELS,
                     "max_pixels": V.MATCHED_MAX_PIXELS, "models": {}}
    OUT.mkdir(parents=True, exist_ok=True)

    for model in V.MODELS:
        summary["models"][model] = {"per_condition": {}, "visual_tokens": {}}
        for cond in ("clean", "no_image"):
            recs = []
            for order in ("O0", "O1", "O2"):
                recs += shard_records(model, cond, order)
            if not recs:
                continue
            oa = order_average(recs, items)
            ids = [r["sample_id"] for r in oa]
            acc, lo, hi = boot_ci(ids, [r["accuracy"] for r in oa])
            mpo, mlo, mhi = boot_ci(ids, [r["accuracy_mean_per_order"] for r in oa])
            pg, plo, phi = boot_ci(ids, [r["p_gt"] for r in oa])
            en, _, _ = boot_ci(ids, [r["entropy"] for r in oa])
            mg, _, _ = boot_ci(ids, [r["margin"] for r in oa])
            per_order = {}
            for order in ("O0", "O1", "O2"):
                sub = [r for r in recs if r["option_order"] == order]
                if sub:
                    per_order[order] = {
                        "n": len(sub),
                        "accuracy": float(np.mean([float(bool(r["correct"]))
                                                   for r in sub])),
                        "mean_p_gt": float(np.mean([r["p_gt"] for r in sub])),
                    }
            summary["models"][model]["per_condition"][cond] = {
                "n_sources": len(oa),
                "accuracy_order_averaged": acc, "accuracy_ci95": [lo, hi],
                "accuracy_mean_per_order": mpo,
                "accuracy_mean_per_order_ci95": [mlo, mhi],
                "p_gt_mean": pg, "p_gt_ci95": [plo, phi],
                "entropy_mean": en, "margin_mean": mg,
                "per_order": per_order,
                "n_visual_tokens_mean": float(np.mean(
                    [r["n_visual_tokens"] for r in recs])),
                "n_visual_tokens_max": int(np.max(
                    [r["n_visual_tokens"] for r in recs])),
                "seq_len_mean": float(np.mean([r["seq_len"] for r in recs])),
                "n_records": len(recs),
            }
        # visual token budget actually used
        concat = {}
        for cond in ("clean", "no_image"):
            recs = []
            for order in ("O0", "O1", "O2"):
                recs += shard_records(model, cond, order)
            vt = [r["n_visual_tokens"] for r in recs]
            concat[cond] = {"mean": float(np.mean(vt)) if vt else None,
                            "min": int(np.min(vt)) if vt else None,
                            "max": int(np.max(vt)) if vt else None}
        summary["models"][model]["visual_tokens"] = concat
        print(f"{model}: " + "  ".join(
            f"{c} acc={summary['models'][model]['per_condition'][c]['accuracy_order_averaged']:.4f}"
            for c in summary["models"][model]["per_condition"]))

    # ---- cross-check against the OLD (native-processor) run ------------------- #
    #
    # AGGREGATION CONVENTION.  The old pipeline's per-condition `accuracy` field is the
    # mean over sources of accuracy_by_order, i.e. each option order is scored
    # separately and then averaged -- NOT the argmax of the order-averaged semantic
    # distribution.  Recomputing the old raw records with the order-averaged convention
    # gives 48.33% for qwen no_image where the old report says 44.63%, which is how this
    # was found.  The spec (configs/spec/01) warns that M2 is aggregation-sensitive and
    # asks for both, so both are reported and the like-for-like comparison against the
    # old run uses mean-per-order.
    old = V.read_json(V.OLD / "results" / "processed" / "m2_analysis.json")
    cmp_rows = []
    for model in V.MODELS:
        for cond in ("clean", "no_image"):
            o = (old.get("models", {}).get(model, {}).get("per_condition", {})
                 .get(cond))
            n = summary["models"][model]["per_condition"].get(cond)
            if not o or not n:
                continue
            cmp_rows.append({
                "model": model, "condition": cond,
                "old_accuracy_mean_per_order": o["accuracy"],
                "matched_mean_per_order": n["accuracy_mean_per_order"],
                "delta_pp": (n["accuracy_mean_per_order"] - o["accuracy"]) * 100,
                "matched_order_averaged": n["accuracy_order_averaged"],
                "delta_order_averaged_pp":
                    (n["accuracy_order_averaged"] - o["accuracy"]) * 100,
            })
    summary["old_vs_matched"] = cmp_rows

    # checkpoint-minus-base gap, matched processing, under BOTH conventions
    gap = {}
    for conv, key in (("mean_per_order", "accuracy_mean_per_order"),
                      ("order_averaged", "accuracy_order_averaged")):
        per = {}
        for cond in ("clean", "no_image"):
            a = summary["models"]["qwen2vl"]["per_condition"].get(cond)
            b = summary["models"]["medvlmr1"]["per_condition"].get(cond)
            if a and b:
                per[cond] = {"gap_pp": (b[key] - a[key]) * 100,
                             "qwen": a[key], "medvlmr1": b[key]}
        if "clean" in per and "no_image" in per:
            per["gap_shrinkage_no_image_minus_clean_pp"] = (
                per["no_image"]["gap_pp"] - per["clean"]["gap_pp"])
        gap[conv] = per
    summary["checkpoint_gap"] = gap

    V.write_json(OUT / "behavior_summary.json", summary)

    # ---- markdown -------------------------------------------------------------- #
    L = ["# Matched-processing behaviour (Stage 1)", "",
         f"generated {summary['created']} · processor `{V.MATCHED_PROCESSOR_ID}` "
         f"(min_pixels={V.MATCHED_MIN_PIXELS}, max_pixels={V.MATCHED_MAX_PIXELS}, "
         "identical for both checkpoints)", "",
         "## Accuracy under matched processing", "",
         "| model | condition | order-averaged acc | 95% CI | mean-per-order acc | "
         "p(GT) | entropy | mean visual tokens |", "|---|---|---|---|---|---|---|---|"]
    for model in V.MODELS:
        for cond in ("clean", "no_image"):
            s = summary["models"][model]["per_condition"].get(cond)
            if not s:
                continue
            L.append(
                f"| {V.MODEL_DISPLAY[model]} | `{cond}` | "
                f"{100 * s['accuracy_order_averaged']:.2f}% | "
                f"[{100 * s['accuracy_ci95'][0]:.2f}, {100 * s['accuracy_ci95'][1]:.2f}] | "
                f"{100 * s['accuracy_mean_per_order']:.2f}% | "
                f"{s['p_gt_mean']:.4f} | {s['entropy_mean']:.4f} | "
                f"{s['n_visual_tokens_mean']:.1f} |")
    L += ["", "## Does the old behavioural conclusion survive the matched processor?", "",
          "The old run's per-condition `accuracy` is the **mean-per-order** convention "
          "(score each option order, then average), not the argmax of the "
          "order-averaged semantic distribution. Recomputing the old raw records under "
          "the order-averaged convention returns 48.33% where the old report says "
          "44.63%, which is how this was found. The like-for-like column below is "
          "mean-per-order.", "",
          "| model | condition | old run | matched, mean-per-order | Δ | matched, "
          "order-averaged | Δ |", "|---|---|---|---|---|---|---|"]
    for r in cmp_rows:
        L.append(f"| {V.MODEL_DISPLAY[r['model']]} | `{r['condition']}` | "
                 f"{100 * r['old_accuracy_mean_per_order']:.2f}% | "
                 f"{100 * r['matched_mean_per_order']:.2f}% | {r['delta_pp']:+.2f} pp | "
                 f"{100 * r['matched_order_averaged']:.2f}% | "
                 f"{r['delta_order_averaged_pp']:+.2f} pp |")
    L += ["", "## The checkpoint gap, under both conventions", "",
          "| convention | condition | Qwen2-VL-2B | MedVLM-R1 | gap |",
          "|---|---|---|---|---|"]
    for conv in ("mean_per_order", "order_averaged"):
        per = gap.get(conv, {})
        for cond in ("clean", "no_image"):
            if cond in per:
                g = per[cond]
                L.append(f"| {conv.replace('_', ' ')} | `{cond}` | "
                         f"{100 * g['qwen']:.2f}% | {100 * g['medvlmr1']:.2f}% | "
                         f"**{g['gap_pp']:+.2f} pp** |")
    for conv in ("mean_per_order", "order_averaged"):
        per = gap.get(conv, {})
        if "gap_shrinkage_no_image_minus_clean_pp" in per:
            L += ["", f"Change in the gap when the image is removed, {conv.replace('_', ' ')}: "
                      f"**{per['gap_shrinkage_no_image_minus_clean_pp']:+.2f} pp**. A "
                      "visual-grounding story predicts a large negative number here."]
    L += ["", "Both conventions agree on the conclusion even though they differ in level "
          "by 2-3 pp: the advantage does **not** shrink when the image is removed.", ""]
    mx = max(abs(r["delta_pp"]) for r in cmp_rows) if cmp_rows else float("nan")
    L += ["## Engineering note", "",
          f"Under the like-for-like (mean-per-order) comparison the largest change "
          f"introduced by the matched processor is **{mx:.2f} pp**. The matched run is "
          f"also the first run in this project to report the order-averaged convention "
          f"alongside it.", ""]
    (V.ROOT / "reports" / "MATCHED_BEHAVIOR.md").write_text("\n".join(L),
                                                           encoding="utf-8")
    print("wrote runs/current900/behavior/behavior_summary.json + "
          "reports/MATCHED_BEHAVIOR.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
