#!/usr/bin/env python
"""Compute the replication metrics required by the task specification.

Produces, per model:
  * overall strict accuracy, format compliance, parse success
  * per-modality accuracy (MRI / CT / X-ray)
  * correct n / total n and a 95% percentile bootstrap CI (10 000 resamples, fixed
    seed)
and across models:
  * a paired breakdown (both correct / only Qwen / only Med / both wrong) with a
    paired bootstrap CI on the accuracy difference and a McNemar test

The accuracy definition is the OFFICIAL one and is not adjusted here:
``correct / len(data)``, so an unparseable answer counts as wrong rather than being
dropped from the denominator. Anything else would inflate the score relative to the
paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Paper reference values (Table 1 of arXiv:2502.19634 for Qwen2-VL-2B and Ours-2B).
PAPER = {
    "qwen2vl": {"overall": 55.11, "MRI": 61.67, "CT": 50.67, "Xray": 53.00},
    "medvlmr1": {"overall": 78.22, "MRI": 95.33, "CT": 70.33, "Xray": 69.00},
}
MODALITIES = ["MRI", "CT", "Xray"]
SEED = 20260913
N_BOOT = 10000


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def bootstrap_ci(flags: np.ndarray, n_boot: int = N_BOOT,
                 seed: int = SEED) -> tuple[float, float]:
    """Percentile bootstrap CI over items (items are the independent unit here:
    each of the 900 has its own image and question)."""
    rng = np.random.default_rng(seed)
    n = len(flags)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = flags[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def mcnemar(b: int, c: int) -> tuple[float, float]:
    """Exact-ish McNemar with continuity correction. Returns (chi2, p)."""
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    # chi2 with 1 df -> p via the normal survival function of sqrt(chi2)
    p = math.erfc(math.sqrt(max(chi2, 0.0) / 2.0))
    return float(chi2), float(p)


def summarise(rows: list[dict]) -> dict:
    out = {}
    flags_all = np.array([bool(r["correct"]) for r in rows])
    out["n_total"] = len(rows)
    out["n_correct"] = int(flags_all.sum())
    out["strict_accuracy"] = float(flags_all.mean())
    out["strict_accuracy_pct"] = float(flags_all.mean() * 100)
    lo, hi = bootstrap_ci(flags_all)
    out["strict_accuracy_ci95"] = [lo, hi]
    out["strict_accuracy_ci95_pct"] = [lo * 100, hi * 100]
    out["parse_success_rate"] = float(np.mean([bool(r["parse_success"]) for r in rows]))
    out["format_compliance_rate"] = float(
        np.mean([bool(r["format_compliant"]) for r in rows]))
    out["mean_generated_tokens"] = float(
        np.mean([r["generated_tokens"] for r in rows]))
    out["n_hit_token_limit"] = int(sum(1 for r in rows if r["generated_tokens"] >= 512))
    out["per_modality"] = {}
    for m in MODALITIES:
        sub = [r for r in rows if r["modality"] == m]
        if not sub:
            continue
        f = np.array([bool(r["correct"]) for r in sub])
        lo_m, hi_m = bootstrap_ci(f)
        out["per_modality"][m] = {
            "n": len(sub),
            "n_correct": int(f.sum()),
            "accuracy": float(f.mean()),
            "accuracy_pct": float(f.mean() * 100),
            "ci95_pct": [lo_m * 100, hi_m * 100],
            "parse_success_rate": float(
                np.mean([bool(r["parse_success"]) for r in sub])),
            "paper": PAPER.get(out.get("model", ""), {}).get(m),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", default=str(ROOT / "results" / "qwen2vl_clean_official900.jsonl"))
    ap.add_argument("--medvlm", default=str(ROOT / "results" / "medvlmr1_clean_official900.jsonl"))
    args = ap.parse_args()

    q = load(Path(args.qwen))
    m = load(Path(args.medvlm))
    print(f"[metrics] qwen={len(q)} medvlm={len(m)}")

    sq = summarise(q); sq["model"] = "qwen2vl"
    sm = summarise(m); sm["model"] = "medvlmr1"
    # model label has to be set before per-modality lookup of paper values
    for s, rows, key in ((sq, q, "qwen2vl"), (sm, m, "medvlmr1")):
        for mod in MODALITIES:
            if mod in s["per_modality"]:
                s["per_modality"][mod]["paper"] = PAPER[key][mod]

    for s, key in ((sq, "qwen2vl"), (sm, "medvlmr1")):
        p = PAPER[key]
        s["paper_overall_pct"] = p["overall"]
        s["delta_vs_paper_pp"] = s["strict_accuracy_pct"] - p["overall"]
        s["per_modality_delta_pp"] = {
            mod: s["per_modality"][mod]["accuracy_pct"] - p[mod]
            for mod in MODALITIES if mod in s["per_modality"]
        }

    # ---- paired comparison on identical items ------------------------------- #
    by_id_q = {r["sample_id"]: r for r in q}
    by_id_m = {r["sample_id"]: r for r in m}
    shared = sorted(set(by_id_q) & set(by_id_m))
    a = np.array([bool(by_id_q[i]["correct"]) for i in shared])   # qwen
    b = np.array([bool(by_id_m[i]["correct"]) for i in shared])   # medvlm
    both = int((a & b).sum())
    only_q = int((a & ~b).sum())
    only_m = int((~a & b).sum())
    neither = int((~a & ~b).sum())
    diff = b.astype(float) - a.astype(float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(shared), size=(N_BOOT, len(shared)))
    boot = diff[idx].mean(axis=1)
    chi2, mcnemar_p = mcnemar(only_q, only_m)

    paired = {
        "n_paired_items": len(shared),
        "both_correct": both,
        "only_qwen_correct": only_q,
        "only_medvlm_correct": only_m,
        "both_wrong": neither,
        "qwen_accuracy": float(a.mean()),
        "medvlm_accuracy": float(b.mean()),
        "accuracy_difference_medvlm_minus_qwen": float(diff.mean()),
        "accuracy_difference_ci95": [float(np.quantile(boot, 0.025)),
                                     float(np.quantile(boot, 0.975))],
        "mcnemar_chi2": chi2,
        "mcnemar_p": mcnemar_p,
        "note": "items are the independent unit; each of the 900 has its own image",
    }

    # ---- per-modality paired ------------------------------------------------ #
    per_mod_paired = {}
    for mod in MODALITIES:
        ids = [i for i in shared if by_id_q[i]["modality"] == mod]
        if not ids:
            continue
        aa = np.array([bool(by_id_q[i]["correct"]) for i in ids])
        bb = np.array([bool(by_id_m[i]["correct"]) for i in ids])
        per_mod_paired[mod] = {
            "n": len(ids),
            "both_correct": int((aa & bb).sum()),
            "only_qwen_correct": int((aa & ~bb).sum()),
            "only_medvlm_correct": int((~aa & bb).sum()),
            "both_wrong": int((~aa & ~bb).sum()),
            "qwen_accuracy_pct": float(aa.mean() * 100),
            "medvlm_accuracy_pct": float(bb.mean() * 100),
            "paper_qwen_pct": PAPER["qwen2vl"][mod],
            "paper_medvlm_pct": PAPER["medvlmr1"][mod],
        }

    # ---- verdict band ------------------------------------------------------- #
    def band(delta):
        return "GREEN" if abs(delta) <= 3 else ("YELLOW" if abs(delta) <= 7 else "RED")
    dq = sq["delta_vs_paper_pp"]; dm = sm["delta_vs_paper_pp"]
    worst = max(abs(dq), abs(dm))
    verdict = ("GREEN" if worst <= 3 else "YELLOW" if worst <= 7 else "RED")
    ordering_ok = sm["strict_accuracy"] > sq["strict_accuracy"]
    gap_ours = sm["strict_accuracy_pct"] - sq["strict_accuracy_pct"]
    gap_paper = PAPER["medvlmr1"]["overall"] - PAPER["qwen2vl"]["overall"]

    report = {
        "qwen2vl": sq, "medvlmr1": sm,
        "paired_comparison": paired,
        "paired_by_modality": per_mod_paired,
        "replication": {
            "band_overall": verdict,
            "band_qwen": band(dq), "band_medvlm": band(dm),
            "delta_qwen_pp": dq, "delta_medvlm_pp": dm,
            "model_ordering_preserved": bool(ordering_ok),
            "gap_ours_pp": gap_ours, "gap_paper_pp": gap_paper,
            "gap_difference_pp": gap_ours - gap_paper,
        },
        "constants": {"seed": SEED, "n_bootstrap": N_BOOT,
                      "accuracy_denominator": "official: correct / len(data); "
                                              "unparsed counts as wrong"},
    }
    out = ROOT / "results" / "replication_metrics.json"
    out.write_text(json.dumps(report, indent=2))

    # ---- summary.csv -------------------------------------------------------- #
    with open(ROOT / "results" / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "scope", "n", "n_correct", "accuracy_pct",
                    "ci95_lo_pct", "ci95_hi_pct", "paper_pct", "delta_pp",
                    "parse_success_rate", "format_compliance_rate"])
        for s, key in ((sq, "qwen2vl"), (sm, "medvlmr1")):
            w.writerow([s["model"], "overall", s["n_total"], s["n_correct"],
                        f"{s['strict_accuracy_pct']:.2f}",
                        f"{s['strict_accuracy_ci95_pct'][0]:.2f}",
                        f"{s['strict_accuracy_ci95_pct'][1]:.2f}",
                        f"{PAPER[key]['overall']:.2f}",
                        f"{s['delta_vs_paper_pp']:+.2f}",
                        f"{s['parse_success_rate']:.4f}",
                        f"{s['format_compliance_rate']:.4f}"])
            for mod in MODALITIES:
                pm = s["per_modality"].get(mod)
                if not pm:
                    continue
                w.writerow([s["model"], mod, pm["n"], pm["n_correct"],
                            f"{pm['accuracy_pct']:.2f}",
                            f"{pm['ci95_pct'][0]:.2f}", f"{pm['ci95_pct'][1]:.2f}",
                            f"{PAPER[key][mod]:.2f}",
                            f"{pm['accuracy_pct'] - PAPER[key][mod]:+.2f}", "", ""])

    with open(ROOT / "results" / "paired_comparison.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scope", "n", "both_correct", "only_qwen_correct",
                    "only_medvlm_correct", "both_wrong",
                    "qwen_accuracy_pct", "medvlm_accuracy_pct",
                    "paper_qwen_pct", "paper_medvlm_pct"])
        w.writerow(["overall", paired["n_paired_items"], both, only_q, only_m, neither,
                    f"{paired['qwen_accuracy']*100:.2f}",
                    f"{paired['medvlm_accuracy']*100:.2f}",
                    f"{PAPER['qwen2vl']['overall']:.2f}",
                    f"{PAPER['medvlmr1']['overall']:.2f}"])
        for mod, v in per_mod_paired.items():
            w.writerow([mod, v["n"], v["both_correct"], v["only_qwen_correct"],
                        v["only_medvlm_correct"], v["both_wrong"],
                        f"{v['qwen_accuracy_pct']:.2f}",
                        f"{v['medvlm_accuracy_pct']:.2f}",
                        f"{v['paper_qwen_pct']:.2f}",
                        f"{v['paper_medvlm_pct']:.2f}"])

    print(f"\n=== REPLICATION: {verdict} ===")
    for s, key in ((sq, "qwen2vl"), (sm, "medvlmr1")):
        p = PAPER[key]
        print(f"  {s['model']:<10} overall {s['strict_accuracy_pct']:6.2f}% "
              f"(paper {p['overall']:.2f}%, {s['delta_vs_paper_pp']:+.2f} pp)  "
              f"per-modality " + " ".join(
                  f"{m}={s['per_modality'][m]['accuracy_pct']:.2f}"
                  f"(p{p[m]:.2f},{s['per_modality_delta_pp'][m]:+.2f})"
                  for m in MODALITIES if m in s["per_modality"]))
    print(f"  paired: both={both} onlyQwen={only_q} onlyMed={only_m} "
          f"neither={neither}  McNemar p={mcnemar_p:.3e}")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
