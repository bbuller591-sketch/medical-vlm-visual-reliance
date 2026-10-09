#!/usr/bin/env python
"""Q3 -- VES-style diagnostic on the task-decision distribution.

THE QUESTION
------------
Joyce asked what an entropy-based visual-evidence sensitivity signal does when the
image is degraded, uninformative, or semantically conflicting.  The interesting
failure is not "VES is bad" but:

        DeltaH > 0   while   U_presence < 0

i.e. adding the image SHARPENS the task-decision distribution while simultaneously
moving it AWAY from the gold answer.  If that happens, an entropy-change reward
cannot on its own tell helpful visual evidence from harmful visual evidence.

DEFINITION
----------
    DeltaH(c)  = H(P_noimage) - H(P_c)      over the normalised A/B/C/D decision
                                            variable, never the full vocabulary
    VES_plus   = max(0, DeltaH)             analysis analogue of the VES reward
    U_presence = M(image) - M(noimage)      signed, in log-margin units
    I_presence = JS(P_c, P_noimage)

Raw DeltaH is primary; the clipped VES_plus is reported alongside because clipping
is exactly what throws away the harmful cases.

CLAIM BOUNDARY (kept in the report itself, not just here)
---------------------------------------------------------
VES-RFT optimises VES together with a verifiable correctness reward.  A diagnostic
showing that entropy change alone does not separate helpful from harmful visual
evidence is NOT a claim that the method is invalid.  The report says only what the
numbers support.
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
from q2_behavior import clustered_boot, load_behavior, paired_boot, js   # noqa: E402

RNG_SEED = 20260917


def entropy(p: dict) -> float:
    return -sum(v * math.log(v) for v in p.values() if v > 0)


# --------------------------------------------------------------------------- #
def unit_tests() -> list[dict]:
    """The spec asks for these explicitly; a diagnostic that cannot fail its own
    sanity checks is not evidence."""
    out = []
    p = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
    q = {0: 0.5, 1: 0.5}
    out.append({"test": "probabilities sum to 1 (4 options)",
                "pass": abs(sum(p.values()) - 1) < 1e-9})
    out.append({"test": "identical distributions give DeltaH = 0",
                "pass": abs((entropy(p) - entropy(p))) < 1e-12})
    # peaked but WRONG: a distribution that is confident about the wrong answer has
    # low entropy, so DeltaH > 0, while the gold margin falls
    noimg = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
    peaked_wrong = {0: 0.97, 1: 0.01, 2: 0.01, 3: 0.01}     # gold = 3
    dh = entropy(noimg) - entropy(peaked_wrong)
    m_no = math.log(noimg[3]) - max(math.log(noimg[j]) for j in (0, 1, 2))
    m_img = math.log(peaked_wrong[3]) - max(math.log(peaked_wrong[j])
                                            for j in (0, 1, 2))
    out.append({"test": "peaked-wrong gives DeltaH > 0 and negative utility",
                "pass": bool(dh > 0 and (m_img - m_no) < 0),
                "value": {"deltaH": round(dh, 4),
                          "U_presence": round(m_img - m_no, 4)}})
    # JS symmetry / identity
    out.append({"test": "JS(p,p) = 0 and JS(p,q) = JS(q,p)",
                "pass": bool(abs(js(p, p)) < 1e-12
                             and abs(js(p, q) - js(q, p)) < 1e-12)})
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen2vl,medvlmr1")
    ap.add_argument("--boot", type=int, default=4000)
    args = ap.parse_args()
    models = [m for m in args.models.split(",") if m]

    out_dir = Q.ROOT / "RESULTS"
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = Q.ROOT / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    tests = unit_tests()
    print("unit tests:", [(t["test"][:40], t["pass"]) for t in tests])
    assert all(t["pass"] for t in tests), "VES unit tests failed"

    beh = load_behavior(models)
    items = Q.load_items()
    ids = sorted(items)
    cl = {s: items[s]["source_image_sha256"] for s in ids}

    rows = []
    for m in models:
        if (m, ids[0], "no_image") not in beh:
            print(f"skip {m}: no no_image arm")
            continue
        for sid in ids:
            n = beh.get((m, sid, "no_image"))
            if n is None:
                continue
            for arm in Q.ARM_ORDER:
                if arm == "no_image":
                    continue
                d = beh.get((m, sid, arm))
                if d is None:
                    continue
                dh = n["H"] - d["H"]
                rows.append({
                    "model": m, "sample_id": sid,
                    "modality": items[sid]["modality"], "arm": arm,
                    "quality_level": Q.ARM_SPEC[arm][0] or "none",
                    "relevance_type": Q.ARM_SPEC[arm][1],
                    "H_noimage": round(n["H"], 6), "H_condition": round(d["H"], 6),
                    "DeltaH": round(dh, 6), "VES_plus": round(max(0.0, dh), 6),
                    "M_noimage": round(n["M"], 6), "M_condition": round(d["M"], 6),
                    "U_presence": round(d["M"] - n["M"], 6),
                    "JS_to_noimage": round(js(d["p"], n["p"]), 6),
                    "correct_noimage": n["acc"] > 0.5, "correct_condition":
                        d["acc"] > 0.5,
                    "harmful_high_sensitivity": bool(dh > 0 and (d["M"] - n["M"]) < 0),
                })
    if not rows:
        print("no rows -- nothing to do yet")
        return
    keys = sorted({k for r in rows for k in r})
    with open(out_dir / "VES_DIAGNOSTIC_PER_ITEM.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    print("wrote VES_DIAGNOSTIC_PER_ITEM.csv", len(rows), "rows")

    # ---------------- aggregates ----------------
    agg = []
    for m in models:
        for arm in Q.ARM_ORDER:
            sub = [r for r in rows if r["model"] == m and r["arm"] == arm]
            if not sub:
                continue
            clus = np.array([cl[r["sample_id"]] for r in sub])
            dh = np.array([r["DeltaH"] for r in sub])
            up = np.array([r["U_presence"] for r in sub])
            jsv = np.array([r["JS_to_noimage"] for r in sub])
            st_dh = clustered_boot(dh, clus, args.boot)
            st_up = clustered_boot(up, clus, args.boot)
            st_js = clustered_boot(jsv, clus, args.boot)
            harmful = np.array([1.0 if (r["DeltaH"] > 0 and r["U_presence"] < 0)
                                else 0.0 for r in sub])
            st_h = clustered_boot(harmful, clus, args.boot)
            pos_dh = dh > 0
            # top-quartile DeltaH with negative utility
            thr = np.percentile(dh, 75)
            top = [r for r in sub if r["DeltaH"] >= thr]
            top_neg = sum(1 for r in top if r["U_presence"] < 0) / max(1, len(top))
            # correlation of DeltaH with U among sharpened cases
            corr = float("nan")
            if pos_dh.sum() > 10:
                a, b = dh[pos_dh], up[pos_dh]
                if a.std() > 0 and b.std() > 0:
                    corr = float(np.corrcoef(a, b)[0, 1])
            agg.append({
                "model": m, "arm": arm,
                "quality_level": Q.ARM_SPEC[arm][0] or "none",
                "relevance_type": Q.ARM_SPEC[arm][1],
                "n": len(sub),
                "mean_DeltaH": round(st_dh["mean"], 4),
                "DeltaH_ci": f"[{st_dh['lo']:.3f},{st_dh['hi']:.3f}]",
                "frac_DeltaH_pos": round(float(pos_dh.mean()), 4),
                "mean_U_presence": round(st_up["mean"], 4),
                "U_ci": f"[{st_up['lo']:.3f},{st_up['hi']:.3f}]",
                "mean_JS": round(st_js["mean"], 5),
                "rate_sharpen_and_harm": round(st_h["mean"], 4),
                "rate_ci": f"[{st_h['lo']:.3f},{st_h['hi']:.3f}]",
                "top_quartile_DeltaH_frac_negative_U": round(top_neg, 4),
                "corr_DeltaH_U_among_sharpened": (None if math.isnan(corr)
                                                  else round(corr, 4)),
            })
    with open(out_dir / "VES_DIAGNOSTIC_SUMMARY.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0].keys()))
        w.writeheader(); w.writerows(agg)
    print("wrote VES_DIAGNOSTIC_SUMMARY.csv", len(agg), "rows")

    # condition-level correlation across samples (spec diagnostic 3)
    corr_rows = []
    for m in models:
        for arm in Q.ARM_ORDER:
            sub = [r for r in rows if r["model"] == m and r["arm"] == arm]
            if len(sub) < 30:
                continue
            dh = np.array([r["DeltaH"] for r in sub])
            up = np.array([r["U_presence"] for r in sub])
            if dh.std() == 0 or up.std() == 0:
                continue
            corr_rows.append({
                "model": m, "arm": arm,
                "quality_level": Q.ARM_SPEC[arm][0] or "none",
                "relevance_type": Q.ARM_SPEC[arm][1], "n": len(sub),
                "corr_DeltaH_U": round(float(np.corrcoef(dh, up)[0, 1]), 4),
                "corr_DeltaH_absU": round(
                    float(np.corrcoef(dh, np.abs(up))[0, 1]), 4)})
    if corr_rows:
        with open(out_dir / "VES_DELTAH_U_CORRELATION.csv", "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(corr_rows[0].keys()))
            w.writeheader(); w.writerows(corr_rows)
        print("wrote VES_DELTAH_U_CORRELATION.csv")

    # ---------------- report ----------------
    lines = [
        "# VES-style diagnostic",
        "",
        "Entropy is computed on the normalised A/B/C/D decision distribution "
        "produced by",
        "the same frozen candidate scorer used for behaviour -- never on the full "
        "vocabulary.",
        "",
        "`DeltaH(c) = H(P_noimage) - H(P_c)`, `VES_plus = max(0, DeltaH)`,",
        "`U_presence(c) = M(c) - M(noimage)` (signed, log-margin units),",
        "`JS_to_noimage(c) = JS(P_c, P_noimage)`.",
        "",
        "## Unit tests",
        ""]
    for t in tests:
        lines.append(f"- [{'PASS' if t['pass'] else 'FAIL'}] {t['test']}"
                     + (f"  `{t['value']}`" if t.get("value") else ""))
    lines += ["", "## Per-arm summary", "",
              "| model | arm | quality | relevance | n | mean DeltaH | "
              "frac DeltaH>0 | mean U_presence | rate sharpened AND harmful | "
              "top-quartile-DeltaH with negative U | corr(DeltaH,U) among "
              "sharpened |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in agg:
        lines.append(
            f"| {a['model']} | {a['arm']} | {a['quality_level']} | "
            f"{a['relevance_type']} | {a['n']} | {a['mean_DeltaH']} "
            f"{a['DeltaH_ci']} | {a['frac_DeltaH_pos']} | {a['mean_U_presence']} "
            f"{a['U_ci']} | {a['rate_sharpen_and_harm']} {a['rate_ci']} | "
            f"{a['top_quartile_DeltaH_frac_negative_U']} | "
            f"{a['corr_DeltaH_U_among_sharpened']} |")
    lines += [
        "",
        "## Claim boundary",
        "",
        "What these numbers can support: that an entropy-change sensitivity signal "
        "does",
        "not by itself separate helpful from harmful visual evidence in this "
        "setting, if",
        "`rate_sharpen_and_harm` is materially above zero.",
        "",
        "What they cannot support: that VES-RFT is invalid.  Its objective pairs "
        "the",
        "entropy term with a verifiable correctness reward, so a diagnostic on one "
        "term",
        "says nothing about the joint objective.",
        ""]
    (rep_dir / "VES_DIAGNOSTIC_RESULTS.md").write_text("\n".join(lines) + "\n",
                                                       encoding="utf-8")
    print("wrote reports/VES_DIAGNOSTIC_RESULTS.md")


if __name__ == "__main__":
    main()
