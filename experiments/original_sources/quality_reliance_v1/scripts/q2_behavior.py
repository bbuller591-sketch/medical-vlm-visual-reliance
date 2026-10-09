#!/usr/bin/env python
"""Q2 -- Quality x Relevance behaviour, utility and the presence/content decomposition.

THE DECOMPOSITION THIS SCRIPT EXISTS TO PRODUCE
------------------------------------------------
For each item, quality level q and donor replicate k:

    M(c)            = log P_c(y*) - max_{j != y*} log P_c(j)        (spec 04)
    U_presence(q)   = M(match_q)        - M(noimage)     image PRESENCE utility
    U_content(q,k)  = M(match_q)        - M(mismatch_q,k) CONTENT utility
    U_mismatch(q,k) = M(mismatch_q,k)   - M(noimage)     harm done by a wrong image
    I_presence(q)   = JS(P_match_q, P_noimage)
    I_content(q,k)  = JS(P_match_q, P_mismatch_qk)

and the identity that makes the whole design cohere:

    U_content(q,k) = U_presence(q) - U_mismatch(q,k)

A large U_content is therefore NOT automatically "more useful vision": it can
come from matched evidence helping, from mismatch evidence harming, or both.  The
decomposition is reported for every cell so the two are never conflated.

U_presence is an image-PRESENCE diagnostic.  Removing the image changes token
presence, sequence length and the multimodal plumbing, so Clean - NoImage is
never described here as content use.  The content quantity is matched - mismatch
at the SAME quality, where both arms receive the same transform.

UNCERTAINTY
-----------
All intervals come from a bootstrap that resamples IMAGE CLUSTERS, not rows.
Several questions in this benchmark share one image; resampling rows would treat
those as independent and understate the interval.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_qr as Q   # noqa: E402

RNG_SEED = 20260917


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_behavior(models: list[str]) -> dict:
    """{(model, sample_id, arm): {'M':.., 'Mp':.., 'acc':.., 'H':.., 'p':{sem:..}}}"""
    out: dict = {}
    for m in models:
        recs = Q.dedupe_load_glob(Q.ROOT / "results" / "raw",
                                  [f"QR__{m}__*.jsonl", f"QR__{m}__*/*.jsonl"])
        by_key: dict = defaultdict(list)
        for r in recs:
            if r.get("status") != "ok":
                continue
            by_key[(r["sample_id"], r["arm"], r["option_order_id"])].append(r)
        # average over option orders (mean-per-order), which is the convention the
        # previous project settled on after two tables were found to disagree
        for (sid, arm, _o), group in by_key.items():
            # for a given order there can be a duplicate row only if a shard was
            # restarted; keep the last, which is what append-only logging implies
            g = group[-1]
            key = (m, sid, arm)
            d = out.setdefault(key, {"margins": [], "margins_p": [], "acc": [],
                                     "H": [], "probs": [], "nviz": [],
                                     "correct": []})
            d["margins"].append(g["gt_log_margin"])
            d["margins_p"].append(g["gt_margin"])
            d["acc"].append(1.0 if g["correct"] else 0.0)
            d["H"].append(g["entropy"])
            d["correct"].append(bool(g["correct"]))
            d["nviz"].append(g.get("n_visual_tokens", 0))
            d["probs"].append({int(k): v for k, v in g["semantic_probs"].items()})
        for (m2, sid, arm), d in list(out.items()):
            if m2 != m:
                continue
            # collapse this model's cells from per-order lists to scalars
            keys = sorted({k for p in d["probs"] for k in p})
            d["M"] = float(np.mean(d["margins"]))
            d["M_p"] = float(np.mean(d["margins_p"]))
            d["acc"] = float(np.mean(d["acc"]))
            d["H"] = float(np.mean(d["H"]))
            d["nviz"] = float(np.mean(d["nviz"]))
            # order-averaged semantic distribution: the marginal over orderings,
            # so JS between two arms is computed on comparable objects
            d["p"] = {k: float(np.mean([p.get(k, 0.0) for p in d["probs"]]))
                       for k in keys}
            d["n_orders"] = len(d["probs"])
    return out


def js(a: dict, b: dict) -> float:
    keys = sorted(set(a) | set(b))
    eps = 1e-12
    pa = np.array([a.get(k, 0.0) for k in keys]) + eps
    pb = np.array([b.get(k, 0.0) for k in keys]) + eps
    pa /= pa.sum(); pb /= pb.sum()
    m = 0.5 * (pa + pb)
    kl = lambda x, y: float(np.sum(x * np.log(x / y)))   # noqa: E731
    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def clustered_boot(values: np.ndarray, clusters: np.ndarray, n_boot: int = 4000,
                   seed: int = RNG_SEED) -> dict:
    """Mean with a percentile CI from resampling CLUSTERS (images), not rows."""
    ok = np.isfinite(values)
    values, clusters = values[ok], clusters[ok]
    if len(values) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0, "n_clusters": 0, "se": float("nan"),
                "frac_positive": float("nan"), "spans_zero": True}
    uniq, inv = np.unique(clusters, return_inverse=True)
    idx = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx[p] for p in pick])
        means[b] = values[sel].mean()
    return {"mean": float(values.mean()),
            "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)),
            "n": int(len(values)), "n_clusters": int(len(uniq)),
            "se": float(means.std(ddof=1))}


def paired_boot(a: np.ndarray, b: np.ndarray, clusters: np.ndarray,
                n_boot: int = 4000, seed: int = RNG_SEED) -> dict:
    """CI for mean(a - b) using ONE resample of clusters for both sides."""
    ok = np.isfinite(a) & np.isfinite(b)
    a, b, clusters = a[ok], b[ok], clusters[ok]
    if len(a) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0, "n_clusters": 0, "se": float("nan"),
                "frac_positive": float("nan"), "spans_zero": True}
    uniq, inv = np.unique(clusters, return_inverse=True)
    idx = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    d = a - b
    means = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx[p] for p in pick])
        means[i] = d[sel].mean()
    return {"mean": float(d.mean()),
            "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)),
            "n": int(len(d)), "n_clusters": int(len(uniq)),
            "se": float(means.std(ddof=1)),
            "frac_positive": float((d > 0).mean()),
            "spans_zero": bool(np.percentile(means, 2.5) <= 0
                               <= np.percentile(means, 97.5))}


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen2vl,medvlmr1")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--out-dir", default="RESULTS")
    args = ap.parse_args()

    models = [m for m in args.models.split(",") if m]
    beh = load_behavior(models)
    items = Q.load_items()
    out_dir = Q.ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = Q.ROOT / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    # cluster = the item's clean image, so questions sharing an image move together
    cl = {sid: items[sid]["source_image_sha256"] for sid in items}
    ids = sorted(items)

    print(f"loaded behaviour for {len(beh)} (model,item,arm) cells")

    # ---------------- per-item tidy table ----------------
    rows = []
    for m in models:
        for sid in ids:
            for arm in Q.ARM_ORDER:
                d = beh.get((m, sid, arm))
                if d is None:
                    continue
                rows.append({
                    "model": m, "sample_id": sid, "modality": items[sid]["modality"],
                    "arm": arm, "quality_level": Q.ARM_SPEC[arm][0] or "none",
                    "relevance_type": Q.ARM_SPEC[arm][1],
                    "n_orders": d["n_orders"],
                    "acc": round(d["acc"], 6), "M_log": round(d["M"], 6),
                    "M_prob": round(d["M_p"], 6), "entropy": round(d["H"], 6),
                    "n_visual_tokens": d["nviz"],
                    "p_correct_semantic": round(d["p"].get(items[sid]["gt_index"],
                                                           float("nan")), 6),
                })
    with open(out_dir / "QUALITY_RELEVANCE_MASTER.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote QUALITY_RELEVANCE_MASTER.csv", len(rows), "rows")

    # ---------------- cohorts ----------------
    # `common_correct` is the intersection of the models' clean-correct sets, so
    # it must be computed before any cohort is requested.
    common = None
    for m in models:
        cc = {sid for sid in ids
              if (beh.get((m, sid, "clean")) or {}).get("acc", 0) > 0.5}
        common = cc if common is None else (common & cc)
    common = common or set()

    def cohort_ids(m: str, name: str) -> list[str]:
        if name == "all":
            return ids
        clean_correct = {sid for sid in ids
                         if (beh.get((m, sid, "clean")) or {}).get("acc", 0) > 0.5}
        if name == "clean_correct":
            return sorted(clean_correct)
        if name == "common_correct":
            return sorted(clean_correct & common)
        raise ValueError(name)

    # ---------------- summary with CIs ----------------
    summary = []
    decomp = []

    def M(m, sid, arm):
        d = beh.get((m, sid, arm))
        return d["M"] if d else float("nan")

    for m in models:
        for cohort in ("all", "clean_correct", "common_correct"):
            sub = cohort_ids(m, cohort)
            if not sub:
                continue
            for stratum in ("ALL", "MRI", "CT", "Xray"):
                s2 = [s for s in sub
                      if stratum == "ALL" or items[s]["modality"] == stratum]
                if not s2:
                    continue
                clus = np.array([cl[s] for s in s2])
                # accuracy and margin per arm
                for arm in Q.ARM_ORDER:
                    v = np.array([M(m, s, arm) for s in s2])
                    a = np.array([(beh.get((m, s, arm)) or {}).get("acc",
                                                                  float("nan"))
                                  for s in s2])
                    stat = clustered_boot(v, clus, args.boot)
                    astat = clustered_boot(a, clus, args.boot)
                    summary.append({
                        "model": m, "cohort": cohort, "stratum": stratum,
                        "quantity": "M_log", "arm": arm,
                        "quality_level": Q.ARM_SPEC[arm][0] or "none",
                        "relevance_type": Q.ARM_SPEC[arm][1],
                        "value": round(stat["mean"], 4),
                        "ci_lo": round(stat["lo"], 4), "ci_hi": round(stat["hi"], 4),
                        "n": stat["n"], "n_clusters": stat["n_clusters"],
                        "accuracy": round(astat["mean"], 4),
                        "acc_ci_lo": round(astat["lo"], 4),
                        "acc_ci_hi": round(astat["hi"], 4)})

                # utility / influence contrasts at each quality level
                for q in Q.QUALITY_LEVELS:
                    ma = Q.MATCHED_ARM[q]
                    mv = np.array([M(m, s, ma) for s in s2])
                    nv = np.array([M(m, s, "no_image") for s in s2])
                    # presence
                    st = paired_boot(mv, nv, clus, args.boot)
                    summary.append({
                        "model": m, "cohort": cohort, "stratum": stratum,
                        "quantity": "U_presence", "arm": f"{ma}_vs_noimage",
                        "quality_level": q, "relevance_type": "contrast",
                        "value": round(st["mean"], 4), "ci_lo": round(st["lo"], 4),
                        "ci_hi": round(st["hi"], 4), "n": st["n"],
                        "n_clusters": st["n_clusters"],
                        "frac_positive": round(st["frac_positive"], 4),
                        "spans_zero": st["spans_zero"]})
                    # influence (JS)
                    pj = np.array([js((beh.get((m, s, ma)) or {}).get("p", {}),
                                      (beh.get((m, s, "no_image")) or {}).get("p", {}))
                                   for s in s2])
                    stj = clustered_boot(pj, clus, args.boot)
                    summary.append({
                        "model": m, "cohort": cohort, "stratum": stratum,
                        "quantity": "I_presence(JS)", "arm": f"{ma}_vs_noimage",
                        "quality_level": q, "relevance_type": "contrast",
                        "value": round(stj["mean"], 5), "ci_lo": round(stj["lo"], 5),
                        "ci_hi": round(stj["hi"], 5), "n": stj["n"],
                        "n_clusters": stj["n_clusters"]})
                    # content, per donor and pooled
                    uc_list, um_list, ui_list = [], [], []
                    sc_pair = (("sc", Q.semantic_arm(q)),)
                    contrasts = [(f"k{k}", Q.mismatch_arm(q, k))
                                 for k in range(1, Q.N_DONORS + 1)]
                    for label, pa in list(contrasts) + list(sc_pair):
                        if not any((m, s, pa) in beh for s in s2[:5]):
                            continue
                        donor_k = int(label[1:]) if label.startswith("k") else None
                        pv = np.array([M(m, s, pa) for s in s2])
                        stc = paired_boot(mv, pv, clus, args.boot)
                        stu = paired_boot(pv, nv, clus, args.boot)
                        pjj = np.array([js((beh.get((m, s, ma)) or {}).get("p", {}),
                                           (beh.get((m, s, pa)) or {}).get("p", {}))
                                        for s in s2])
                        sti = clustered_boot(pjj, clus, args.boot)
                        if label.startswith("k"):
                            uc_list.append(stc["mean"]); um_list.append(stu["mean"])
                            ui_list.append(sti["mean"])
                        summary.append({
                            "model": m, "cohort": cohort, "stratum": stratum,
                            "quantity": "U_content", "arm": f"{ma}_vs_{pa}",
                            "donor_k": donor_k, "contrast_family":
                                "semantic_conflict" if label == "sc"
                                else "plausible_mismatch",
                            "quality_level": q,
                            "relevance_type": "contrast",
                            "value": round(stc["mean"], 4),
                            "ci_lo": round(stc["lo"], 4),
                            "ci_hi": round(stc["hi"], 4), "n": stc["n"],
                            "n_clusters": stc["n_clusters"],
                            "frac_positive": round(stc["frac_positive"], 4),
                            "spans_zero": stc["spans_zero"]})
                        summary.append({
                            "model": m, "cohort": cohort, "stratum": stratum,
                            "quantity": "U_mismatch", "arm": f"{pa}_vs_noimage",
                            "donor_k": donor_k, "contrast_family":
                                "semantic_conflict" if label == "sc"
                                else "plausible_mismatch",
                            "quality_level": q,
                            "relevance_type": "contrast",
                            "value": round(stu["mean"], 4),
                            "ci_lo": round(stu["lo"], 4),
                            "ci_hi": round(stu["hi"], 4), "n": stu["n"],
                            "n_clusters": stu["n_clusters"],
                            "frac_positive": round(stu["frac_positive"], 4),
                            "spans_zero": stu["spans_zero"]})
                        summary.append({
                            "model": m, "cohort": cohort, "stratum": stratum,
                            "quantity": "I_content(JS)", "arm": f"{ma}_vs_{pa}",
                            "donor_k": donor_k, "contrast_family":
                                "semantic_conflict" if label == "sc"
                                else "plausible_mismatch",
                            "quality_level": q,
                            "relevance_type": "contrast",
                            "value": round(sti["mean"], 5),
                            "ci_lo": round(sti["lo"], 5),
                            "ci_hi": round(sti["hi"], 5), "n": sti["n"],
                            "n_clusters": sti["n_clusters"]})
                    if uc_list:
                        decomp.append({
                            "model": m, "cohort": cohort, "stratum": stratum,
                            "quality_level": q,
                            "n": int(st["n"]),
                            "U_presence": round(st["mean"], 4),
                            "U_content_mean": round(float(np.mean(uc_list)), 4),
                            "U_content_sd_across_donors":
                                round(float(np.std(uc_list, ddof=1))
                                      if len(uc_list) > 1 else 0.0, 4),
                            "U_mismatch_mean": round(float(np.mean(um_list)), 4),
                            "U_mismatch_sd_across_donors":
                                round(float(np.std(um_list, ddof=1))
                                      if len(um_list) > 1 else 0.0, 4),
                            "I_content_mean": round(float(np.mean(ui_list)), 5),
                            "identity_residual": round(
                                float(np.mean(uc_list))
                                - (st["mean"] - float(np.mean(um_list))), 6),
                            "harmful_mismatch_rate":
                                round(float(np.mean([
                                    (beh.get((m, s, Q.mismatch_arm(q, k)))
                                     or {}).get("M", np.nan)
                                    - nv[i] < 0
                                    for k in range(1, Q.N_DONORS + 1)
                                    for i, s in enumerate(s2)])), 4)})

    def dump(rows_, name):
        if not rows_:
            print("skip (empty):", name)
            return
        keys = sorted({k for r in rows_ for k in r})
        with open(out_dir / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows_:
                w.writerow(r)
        print("wrote", name, len(rows_), "rows")

    dump(summary, "QUALITY_RELEVANCE_SUMMARY.csv")
    dump(decomp, "QUALITY_RELEVANCE_DECOMPOSITION.csv")

    # ---------------- markdown ----------------
    lines = ["# Quality x Relevance results", "",
             "Generated by `scripts/q2_behavior.py`.  M is the spec-04 log margin",
             "`log P(y*) - max_{j!=y*} log P(j)`; U are differences of M; I are JS",
             "divergences.  Intervals are percentile CIs from a cluster bootstrap",
             f"({args.boot} resamples) clustering on the item's clean image.", "",
             "`U_content = U_presence - U_mismatch` holds per item; the table below",
             "reports the residual of that identity as a check.", ""]
    lines += ["| model | stratum | quality | U_presence | U_content (mean) | "
              "U_content sd across donors | U_mismatch | identity resid | "
              "harmful mismatch rate |",
              "|---|---|---|---|---|---|---|---|---|"]
    for d in decomp:
        lines.append("| {model} | {stratum} | {quality_level} | {U_presence} | "
                     "{U_content_mean} | {U_content_sd_across_donors} | "
                     "{U_mismatch_mean} | {identity_residual} | "
                     "{harmful_mismatch_rate} |".format(**d))
    (rep_dir / "QUALITY_RELEVANCE_RESULTS.md").write_text("\n".join(lines) + "\n",
                                                          encoding="utf-8")
    print("wrote reports/QUALITY_RELEVANCE_RESULTS.md")

    Q.write_json(Q.ROOT / "runtime" / "q2_summary.json", {
        "n_cells": len(beh), "n_master_rows": len(rows),
        "n_summary_rows": len(summary), "n_decomp_rows": len(decomp),
        "common_correct_n": len(common),
        "models": models, "time": Q.now()})


if __name__ == "__main__":
    main()
