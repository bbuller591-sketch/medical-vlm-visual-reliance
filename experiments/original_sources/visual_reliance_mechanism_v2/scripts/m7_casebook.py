#!/usr/bin/env python
"""Stage 6 -- causal casebook selection (spec 04 Stage 6).

Selection is based on **pre-existing behaviour only** (clean / no-image / mismatch
outcomes), never on mechanism outcomes, so the causal tests that follow are not
selected on the thing they measure.

Four phenotypes, 25 sources each where available:

1. `med_prior_dominant_mri`  MRI items where MedVLM-R1 gets the right answer with NO
   image at high margin while Qwen2-VL also gets it wrong with no image, and clean
   stays correct for MedVLM-R1.  This is the "MRI advantage without the image" case.
2. `visually_corrected`       clean correct but no-image wrong (the image supplies the
   answer) for the checkpoint.
3. `mismatch_sensitive`       clean correct with a healthy margin, but the
   random-same-modality replacement flips the answer or destroys the margin.
4. `stable_control`           clean, no-image and mismatch all agree with similar
   margin for the checkpoint -- the negative control for any patch effect.

Writes causal/casebook.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v2 as V      # noqa: E402
import featlib as F        # noqa: E402

OUT = V.ROOT / "causal"
TARGET = 25
BASE, CHK = "qwen2vl", "medvlmr1"


def per_sample(model: str, cond: str) -> dict:
    """sample_id -> {correct, p_gt, margin} averaged over the distinct option orders."""
    if not F.has(model, cond):
        return {}
    recs = F.load_meta(model, cond)
    by = defaultdict(list)
    for r in recs:
        by[r["sample_id"]].append(r)
    out = {}
    for sid, rs in by.items():
        seen = {}
        for r in rs:
            seen[tuple(r["permutation"])] = r
        vals = list(seen.values())
        n = vals[0]["n_options"]
        acc = np.zeros(n)
        for r in vals:
            acc += np.array([r["semantic_probs"].get(str(i), 0.0) for i in range(n)])
        pm = acc / acc.sum()
        out[sid] = {
            "correct": float(int(np.argmax(pm)) == vals[0]["gt_semantic"]),
            "p_gt": float(pm[vals[0]["gt_semantic"]]),
            "n_orders": len(vals),
            "top_semantic": int(np.argmax(pm)),
        }
    return out


def main() -> int:
    items = V.load_items()
    beh = {}
    for model in (BASE, CHK):
        beh[model] = {c: per_sample(model, c) for c in
                      ("clean", "no_image", "random_same_modality_replacement",
                       "opposite_answer_donor", "same_answer_donor")}

    b, c = beh[BASE], beh[CHK]
    pools = {"med_prior_dominant_mri": [], "visually_corrected": [],
             "mismatch_sensitive": [], "stable_control": []}
    detail = {}
    for sid, it in items.items():
        if not all(sid in c[x] for x in ("clean", "no_image",
                                         "random_same_modality_replacement")):
            continue
        if not all(sid in b[x] for x in ("clean", "no_image")):
            continue
        ccl, cno, crm = c["clean"][sid], c["no_image"][sid], \
            c["random_same_modality_replacement"][sid]
        bcl, bno = b["clean"][sid], b["no_image"][sid]
        rec = {
            "sample_id": sid, "modality": it["modality"], "n_options": it["n_options"],
            "ckpt_clean_correct": ccl["correct"], "ckpt_clean_p_gt": ccl["p_gt"],
            "ckpt_no_image_correct": cno["correct"], "ckpt_no_image_p_gt": cno["p_gt"],
            "ckpt_mismatch_correct": crm["correct"], "ckpt_mismatch_p_gt": crm["p_gt"],
            "base_clean_correct": bcl["correct"], "base_no_image_correct": bno["correct"],
            "base_no_image_p_gt": bno["p_gt"],
            "margin_drop_clean_to_mismatch": ccl["p_gt"] - crm["p_gt"],
            "margin_gain_no_image_to_clean": ccl["p_gt"] - cno["p_gt"],
        }
        detail[sid] = rec
        if (it["modality"] == "MRI" and cno["correct"] == 1.0 and cno["p_gt"] >= 0.6
                and bno["correct"] == 0.0 and ccl["correct"] == 1.0):
            pools["med_prior_dominant_mri"].append(sid)
        if ccl["correct"] == 1.0 and cno["correct"] == 0.0:
            pools["visually_corrected"].append(sid)
        if (ccl["correct"] == 1.0 and ccl["p_gt"] >= 0.6
                and (crm["correct"] == 0.0 or crm["p_gt"] <= 0.25)):
            pools["mismatch_sensitive"].append(sid)
        if (ccl["correct"] == cno["correct"] == crm["correct"] == 1.0
                and abs(ccl["p_gt"] - cno["p_gt"]) <= 0.15
                and abs(ccl["p_gt"] - crm["p_gt"]) <= 0.15):
            pools["stable_control"].append(sid)

    rng = np.random.default_rng(V.GLOBAL_SEED)
    selected, counts = {}, {}
    for name, ids in pools.items():
        ids = sorted(ids)
        counts[name] = {"available": len(ids), "selected": min(TARGET, len(ids))}
        if len(ids) <= TARGET:
            selected[name] = ids
        else:
            pick = rng.choice(len(ids), size=TARGET, replace=False)
            selected[name] = [ids[int(i)] for i in sorted(pick)]

    casebook = {
        "created": V.now(),
        "rule": ("selection uses pre-existing behaviour only (clean / no_image / "
                 "mismatch outcomes), never mechanism outcomes"),
        "target_per_phenotype": TARGET,
        "counts": counts,
        "phenotypes": selected,
        "detail": {sid: detail[sid] for ids in selected.values() for sid in ids},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    V.write_json(OUT / "casebook.json", casebook)

    L = ["# Causal casebook (Stage 6)", "",
         f"generated {casebook['created']}", "",
         "Selected on **pre-existing behaviour only**, so the causal tests that follow "
         "are not chosen on the quantity they measure.", "",
         "| phenotype | available | selected |", "|---|---|---|"]
    for k, v in counts.items():
        L.append(f"| `{k}` | {v['available']} | {v['selected']} |")
    L += ["", "| phenotype | what it means |", "|---|---|",
          "| `med_prior_dominant_mri` | MRI; MedVLM-R1 correct with no image at "
          "p(GT)>=0.6 while Qwen2-VL is wrong without an image; checkpoint clean still "
          "correct (the MRI advantage that does not need the image) |",
          "| `visually_corrected` | checkpoint clean correct, no-image wrong — the "
          "image supplies the answer |",
          "| `mismatch_sensitive` | checkpoint clean correct with p(GT)>=0.6, but "
          "random same-modality replacement flips it or drops p(GT)<=0.25 |",
          "| `stable_control` | clean, no-image and mismatch all agree within 0.15 "
          "p(GT) — the null case for any patch effect |", ""]
    for name, ids in selected.items():
        L += [f"## `{name}` ({len(ids)})", "",
              ", ".join(f"`{x}`" for x in ids), ""]
    (V.ROOT / "reports" / "CASEBOOK.md").write_text("\n".join(L), encoding="utf-8")

    print(json.dumps(counts, indent=1))
    print(f"wrote causal/casebook.json ({sum(len(v) for v in selected.values())} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
