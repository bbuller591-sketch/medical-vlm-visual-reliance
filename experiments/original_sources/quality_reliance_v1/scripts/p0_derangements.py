#!/usr/bin/env python
"""P0-B -- build K=3 plausible-mismatch derangements that are CLOSED UNDER FOLDS.

WHY A DERANGEMENT, AND WHY IT MUST BE BUILT IN CLOSED SUBGROUPS
--------------------------------------------------------------
Two properties are needed, and the second is the one that is easy to get wrong.

1. Plausible mismatch.  A donor must be same-modality, never the item's own
   image, and not a duplicate.  A random draw satisfies this.

2. Balance on every fold that will actually be used.
   Item i's matched arm shows image(i); its mismatch arm shows image(donor(i)).
   Over a set S, the matched class shows {image(i) : i in S} and the mismatch
   class shows {image(donor(i)) : i in S}.  These are equal as multisets if and
   only if donor(.) maps S onto S.  A probe trained on a cross-validation fold
   sees only that fold's training set, so the balance has to hold on the TRAINING
   SUBSET, not just on all 900.

   A single cycle over a whole modality is a bijection of the whole modality but
   of no proper subset of it, so any fold smaller than the modality breaks the
   balance.  A probe can then recover "matched or mismatch" from image statistics
   alone, and the control stops being a control -- the previous round of this
   project shipped exactly that bug, with a control that had to read 0.500
   reporting 0.057.

   The construction here deranges WITHIN small subgroups and records the subgroup
   id.  Every subgroup is closed under its own derangement, so every union of
   subgroups is closed too, and any fold built as a union of subgroups has exactly
   balanced matched and mismatch image multisets.  That is a structural guarantee
   rather than a hope, and `--verify` proves it on the exact subsets the probes
   will use.

Output: manifests/donor_derangements.json
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_qr as Q   # noqa: E402

SUBGROUP_SIZE = 30
NEAR_DUP_CORR = 0.9995


def thumb(path: Path, n: int = 32) -> np.ndarray | None:
    """Small greyscale thumbnail, for near-duplicate detection only."""
    try:
        from PIL import Image
        im = Image.open(path).convert("L").resize((n, n))
        a = np.asarray(im, dtype=np.float64)
        a = a - a.mean()
        s = a.std()
        return a / s if s > 1e-9 else np.zeros_like(a)
    except Exception:
        return None


def build_subgroups(ids: list[str], items: dict, seed_key: str,
                    size: int) -> list[list[str]]:
    """Partition within modality into near-equal closed subgroups."""
    rng = random.Random(Q.stable_seed(seed_key, "subgroup"))
    out: list[list[str]] = []
    by_mod: dict[str, list[str]] = defaultdict(list)
    for s in ids:
        by_mod[items[s]["modality"]].append(s)
    for mod in sorted(by_mod):
        group = sorted(by_mod[mod])
        rng.shuffle(group)
        chunks = [group[i:i + size] for i in range(0, len(group), size)]
        # a chunk of one or two items has too few distinct images to derange
        # internally, so a small tail is folded into the previous chunk
        if len(chunks) > 1 and len(chunks[-1]) < 3:
            chunks[-2].extend(chunks.pop())
        out.extend(chunks)
    return out


def derange_within(sub: list[str], seed_key: str, bad_pairs: set) -> dict[str, str]:
    """A fixed-point-free bijection of `sub` avoiding `bad_pairs` when possible."""
    rng = random.Random(Q.stable_seed(seed_key, "shift", sub[0]))
    n = len(sub)
    order = sub[:]
    for _ in range(4000):
        rng.shuffle(order)
        shift = rng.randrange(1, n)
        rot = order[shift:] + order[:shift]
        if all(k != v and (k, v) not in bad_pairs for k, v in zip(order, rot)):
            return dict(zip(order, rot))
    for _ in range(4000):            # relax duplicate avoidance, repair handles it
        rng.shuffle(order)
        shift = rng.randrange(1, n)
        rot = order[shift:] + order[:shift]
        if all(k != v for k, v in zip(order, rot)):
            return dict(zip(order, rot))
    raise RuntimeError("could not derange subgroup")


def repair(assignment: dict, sub: set, bad_pairs: set) -> tuple[dict, int]:
    """Swap donors between two targets until no bad pair remains.

    A swap preserves the bijection because it exchanges two targets' images.
    """
    ids = sorted(sub)
    n_fix = 0
    for _ in range(200):
        bad = [(t, d) for t, d in assignment.items()
               if (t, d) in bad_pairs or t == d]
        if not bad:
            break
        done = False
        for t, _d in bad:
            for t2 in ids:
                if t2 == t:
                    continue
                a2 = dict(assignment)
                a2[t], a2[t2] = assignment[t2], assignment[t]
                before = sum(1 for x, y in assignment.items()
                             if (x, y) in bad_pairs or x == y)
                after = sum(1 for x, y in a2.items()
                            if (x, y) in bad_pairs or x == y)
                if after < before:
                    assignment = a2
                    n_fix += 1
                    done = True
                    break
            if done:
                break
        if not done:
            break
    return assignment, n_fix


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-base", default="quality_reliance_v1/derangement_v2")
    ap.add_argument("--k", type=int, default=Q.N_DONORS)
    ap.add_argument("--subgroup-size", type=int, default=SUBGROUP_SIZE)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    items = Q.load_items()
    ids = sorted(items)
    sha = {s: items[s]["source_image_sha256"] for s in ids}

    print("computing thumbnails for near-duplicate detection ...")
    th = {}
    for s in ids:
        t = thumb(Q.source_image_path(items[s]))
        if t is not None:
            th[s] = t
    print(f"  {len(th)}/{len(ids)} thumbnails")

    out = {"seed_base": args.seed_base, "n_items": len(ids),
           "subgroup_size": args.subgroup_size,
           "near_dup_corr_threshold": NEAR_DUP_CORR, "derangements": {}}

    for k in range(1, args.k + 1):
        subs = build_subgroups(ids, items, f"{args.seed_base}/D{k}",
                               args.subgroup_size)
        assignment: dict[str, str] = {}
        subgroup_of: dict[str, int] = {}
        n_dup_repaired = 0
        for gi, sub in enumerate(subs):
            bad = {(a, b) for a in sub for b in sub
                   if a != b and sha[a] == sha[b]}
            for i, a in enumerate(sub):          # near-duplicates, in-subgroup only
                if a not in th:
                    continue
                for b in sub[i + 1:]:
                    if b not in th:
                        continue
                    if float(np.mean(th[a] * th[b])) > NEAR_DUP_CORR:
                        bad.add((a, b)); bad.add((b, a))
            a0 = derange_within(sub, f"{args.seed_base}/D{k}/g{gi}", bad)
            a0, nf = repair(a0, set(sub), bad)
            n_dup_repaired += nf
            assignment.update(a0)
            for x in sub:
                subgroup_of[x] = gi
        assert len(set(assignment.values())) == len(ids), "not a bijection"

        diag = {
            "n": len(ids),
            "fixed_points": sum(1 for t, d in assignment.items() if t == d),
            "same_image_sha": sum(1 for t, d in assignment.items()
                                  if sha[t] == sha[d]),
            "cross_modality": sum(1 for t, d in assignment.items()
                                  if items[t]["modality"] != items[d]["modality"]),
            "cross_subgroup": sum(1 for t, d in assignment.items()
                                  if subgroup_of[t] != subgroup_of[d]),
            "n_subgroups": len(subs),
            "subgroup_size_histogram": {str(a): b for a, b in sorted(
                Counter(len(s) for s in subs).items())},
            "near_dup_repaired_by_swap": n_dup_repaired,
            "fraction_same_gt_text": round(sum(
                items[t].get("gt_norm") == items[d].get("gt_norm")
                for t, d in assignment.items()) / len(ids), 4),
            "fraction_same_option_set": round(sum(
                items[t].get("option_set_key") == items[d].get("option_set_key")
                for t, d in assignment.items()) / len(ids), 4),
            "per_modality_same_gt": {
                m: round(sum(1 for t in ids if items[t]["modality"] == m
                             and items[t].get("gt_norm")
                             == items[assignment[t]].get("gt_norm"))
                         / max(1, sum(1 for t in ids
                                      if items[t]["modality"] == m)), 4)
                for m in sorted({items[t]["modality"] for t in ids})},
        }

        if not args.no_verify:
            # THE balance guarantee, checked on the exact subsets the probes use
            for gi in sorted(set(subgroup_of.values())):
                S = [t for t in ids if subgroup_of[t] == gi]
                if Counter(sha[t] for t in S) != Counter(sha[assignment[t]]
                                                         for t in S):
                    raise AssertionError(
                        f"D{k} subgroup {gi} is not balanced: the matched and "
                        f"mismatch image multisets differ, which is what lets a "
                        f"vision-only probe beat chance")
            rng = random.Random(Q.stable_seed(args.seed_base, "folds"))
            order = sorted(set(subgroup_of.values()))
            rng.shuffle(order)
            for f in range(5):
                S = [t for t in ids if subgroup_of[t] in set(order[f::5])]
                if S and Counter(sha[t] for t in S) != Counter(
                        sha[assignment[t]] for t in S):
                    raise AssertionError(f"D{k} 5-fold split {f} is not balanced")
        diag["balance_verified"] = not args.no_verify

        out["derangements"][f"D{k}"] = {"assignment": assignment,
                                        "subgroup_of": subgroup_of, "diag": diag}
        print(f"D{k}: fixed={diag['fixed_points']} same_sha={diag['same_image_sha']}"
              f" cross_sub={diag['cross_subgroup']}"
              f" subgroups={diag['n_subgroups']}"
              f" same_gt={diag['fraction_same_gt_text']:.3f}"
              f" verified={diag['balance_verified']}")

    keys = [f"D{k}" for k in range(1, args.k + 1)]
    agreed = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = out["derangements"][keys[i]]["assignment"]
            b = out["derangements"][keys[j]]["assignment"]
            agreed[f"{keys[i]}_{keys[j]}"] = round(
                sum(1 for t in ids if a[t] == b[t]) / len(ids), 4)
    out["pairwise_donor_agreement"] = agreed
    out["folds"] = ("any union of subgroups; matched/mismatch image multisets are "
                    "then identical by construction")
    print("pairwise donor agreement:", agreed)

    Q.write_json(Q.ROOT / "manifests" / "donor_derangements.json", out)
    print("written: manifests/donor_derangements.json")


if __name__ == "__main__":
    main()
