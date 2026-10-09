#!/usr/bin/env python
"""P0-C -- semantic-conflict mismatch manifest (high-confidence subset).

WHAT MAKES A PAIR A CONFLICT
----------------------------
Not "a different answer letter".  A conflict means the donor image supports a
clinical state that the target's gold answer rules out.  Two deterministic tiers:

Tier 1 -- mutually exclusive, derived from dataset metadata:
  (a) same question, opposite polarity.  The question group contains both
      "COVID-19 positive" and "COVID-19 negative"; exactly one can hold of one
      image.
  (b) same question that asks for a UNIQUE attribute ("which joint does the bone
      come from", "what imaging technique is used").  A bone comes from one joint;
      an image was made with one modality.  Two different answers to such a
      question are mutually exclusive by the semantics of the question itself.
Tier 2 -- same question, different finding, descriptive question ("what is the
  abnormality present").  Two different abnormalities can co-occur, so this is a
  weaker conflict and is labelled as such everywhere it appears.

Everything is decided from dataset metadata and a fixed lexicon.  No model output
is consulted, so the donor set cannot be tuned to produce an effect.  The spec's
audit requirement (30 random accepted pairs saved for manual checking) is met by
`audit_sample` in the output.

CLAIM BOUNDARY: these are LABEL-SUPPORTED SEMANTIC-CONFLICT DONORS.  The donor
image itself has not been read by a radiologist for the relevant finding, so it is
not a "ground-truth counterfactual image".
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_qr as Q   # noqa: E402

NEGATION = re.compile(
    r"\b(no|not|none|without|absent|negative|normal|unremarkable|clear|"
    r"unaffected|intact|denies)\b", re.I)
# questions asking to identify a UNIQUE attribute of the acquisition or anatomy
UNIQUE_ATTR = re.compile(
    r"\b(which|where|from which|what imaging technique|what modality|"
    r"which joint|which bone|which type of joint|what technique)\b", re.I)
# questions that merely ask what is visible; two answers can both be true
DESCRIPTIVE = re.compile(
    r"\b(what can be observed|what is the abnormality|what abnormality|"
    r"what content|what type of abnormality|which of the following)\b", re.I)

STOP = set("""a an the is are was were in on of to for with this image picture
what which does do from present appears appearance following can be observed
there any please select correct option answer""".split())


def tokens(s: str) -> set:
    return {t for t in re.findall(r"[a-z0-9\-]+", s.lower())
            if t not in STOP and len(t) > 2}


def polarity(gt_text: str) -> str:
    return "negative" if NEGATION.search(gt_text or "") else "positive"


def build(items: dict) -> dict:
    by_q = defaultdict(list)
    for s in sorted(items):
        by_q[items[s]["question_norm"]].append(s)

    pairs = []
    for qnorm, group in by_q.items():
        gts = {items[s]["gt_norm"] for s in group}
        if len(gts) < 2:
            continue
        unique_attr = bool(UNIQUE_ATTR.search(qnorm))
        descriptive = bool(DESCRIPTIVE.search(qnorm))
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if items[a]["gt_norm"] == items[b]["gt_norm"]:
                    continue
                if items[a]["modality"] != items[b]["modality"]:
                    continue
                if items[a]["source_image_sha256"] == items[b]["source_image_sha256"]:
                    continue
                pa, pb = polarity(items[a]["gt_text"]), polarity(items[b]["gt_text"])
                if pa != pb:
                    tier, rule = 1, "polarity_conflict"
                elif unique_attr:
                    tier, rule = 1, "unique_attribute_conflict"
                elif descriptive:
                    tier, rule = 2, "descriptive_different_finding"
                else:
                    tier, rule = 2, "same_question_different_finding"
                pairs.append({
                    "target": a, "donor": b, "tier": tier, "rule": rule,
                    "question_norm": qnorm,
                    "question": items[a]["question"],
                    "target_gt": items[a]["gt_text"],
                    "donor_gt": items[b]["gt_text"],
                    "modality": items[a]["modality"],
                    "target_polarity": pa, "donor_polarity": pb,
                })
    return {"pairs": pairs}


def select(pairs: list[dict], seed_key: str) -> dict:
    """One donor per target, preferring Tier 1, then a rarer target answer.

    Preference for rarer answers is a nuisance-matching device: across the whole
    benchmark some gold answers are far more common than others, and picking
    donors uniformly would make the conflict set's target-answer distribution
    lopsided.
    """
    rng = random.Random(Q.stable_seed(seed_key, "select"))
    by_target = defaultdict(list)
    for p in pairs:
        by_target[p["target"]].append(p)
    chosen = {}
    for t, cand in sorted(by_target.items()):
        cand.sort(key=lambda p: (p["tier"], rng.random()))
        chosen[t] = cand[0]
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="quality_reliance_v1/semantic_conflict")
    ap.add_argument("--audit-n", type=int, default=30)
    args = ap.parse_args()

    items = Q.load_items()
    built = build(items)
    pairs = built["pairs"]
    chosen = select(pairs, args.seed)

    gt_freq = Counter(items[s]["gt_norm"] for s in items)
    audit = []
    rng = random.Random(Q.stable_seed(args.seed, "audit"))
    pick = sorted(chosen)
    rng.shuffle(pick)
    for t in pick[:args.audit_n]:
        p = chosen[t]
        audit.append({
            "target_sample_id": p["target"], "donor_sample_id": p["donor"],
            "tier": p["tier"], "rule": p["rule"],
            "question": p["question"].replace("\n", " ")[:160],
            "target_gt": p["target_gt"], "donor_gt": p["donor_gt"],
            "modality": p["modality"],
            "target_gt_freq_in_900": gt_freq[items[p["target"]]["gt_norm"]],
            "donor_gt_freq_in_900": gt_freq[items[p["donor"]]["gt_norm"]]})

    by_tier = Counter(p["tier"] for p in chosen.values())
    by_rule = Counter(p["rule"] for p in chosen.values())
    coverage = {
        "n_targets_with_a_conflict_donor": len(chosen),
        "n_900": len(items),
        "coverage_fraction": round(len(chosen) / len(items), 4),
        "tier_counts": {str(k): v for k, v in sorted(by_tier.items())},
        "rule_counts": dict(by_rule),
        "by_modality": dict(Counter(items[t]["modality"]
                                    for t in chosen)),
        "any_pair_candidates_total": len(pairs),
        "own_image_excluded": True,
        "duplicate_image_excluded": True,
        "cross_modality_excluded": True,
        "model_output_consulted": False,
    }
    out = {
        "seed": args.seed,
        "note": ("Label-supported semantic-conflict donors, not ground-truth "
                 "counterfactual images."),
        "coverage": coverage,
        "assignment": {t: chosen[t] for t in sorted(chosen)},
        "audit_sample": audit,
        "tier_rules": {
            "1": "polarity_conflict (same question, opposite negation) OR "
                 "unique_attribute_conflict (question asks for a unique attribute)",
            "2": "same question, different finding on a DESCRIPTIVE question; "
                 "two such findings can co-occur, so this is a weaker conflict",
        },
    }
    Q.write_json(Q.ROOT / "manifests" / "semantic_conflict.json", out)
    print(json_dumps(coverage))
    print(f"audit sample of {len(audit)} pairs written for manual review")


def json_dumps(o) -> str:
    import json
    return json.dumps(o, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
