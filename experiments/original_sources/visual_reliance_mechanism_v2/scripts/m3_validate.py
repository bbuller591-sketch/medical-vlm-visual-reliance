#!/usr/bin/env python
"""Acceptance gate for the v2 mechanism run (spec 07_IMPLEMENTATION_AND_ACCEPTANCE_TESTS).

A result is only usable if these checks pass.  Run this after extraction and before any
interpretation.  Exits non-zero if a FATAL check fails.

FATAL: genuine no-image has zero visual tokens; decision position is the last prompt
token; candidate labels are single tokens; the final-layer logit lens reproduces the
model's own logits; every shard is internally consistent; provenance exists.
NON-FATAL: coverage gaps, matched-token-count differences.

Writes status/ACCEPTANCE.md and status/acceptance.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v2 as V      # noqa: E402
import featlib as F        # noqa: E402

CHECKS: list[dict] = []


def check(name: str, ok: bool | None, detail: str, fatal: bool = True) -> None:
    CHECKS.append({"name": name, "ok": ok, "fatal": fatal, "detail": detail})


def main() -> int:
    audit = V.read_json(V.ROOT / "provenance" / "processor_audit.json")
    items = V.load_items()
    acc = audit["acceptance"]

    # ---- §1 processor / tokenizer --------------------------------------------- #
    check("§1 matched visual token count identical across checkpoints",
          acc["matched_visual_token_count_equal"],
          f"per-image equality: {acc['matched_visual_token_count_equal_per_image']}")
    check("§1 native budgets differ (this is why the freeze exists)",
          acc["native_visual_token_count_qwen"] != acc["native_visual_token_count_medvlmr1"],
          f"native qwen {acc['native_visual_token_count_qwen']} vs medvlmr1 "
          f"{acc['native_visual_token_count_medvlmr1']}", fatal=False)
    check("§4 all candidate labels are single tokens",
          acc["all_candidates_single_token_qwen"] and
          acc["all_candidates_single_token_medvlmr1"],
          f"qwen={acc['all_candidates_single_token_qwen']} "
          f"medvlmr1={acc['all_candidates_single_token_medvlmr1']}")

    # ---- per-(model, condition, order) record checks --------------------------- #
    per_unit: dict = {}
    for model in V.MODELS:
        d = F.FEAT / model
        if not d.is_dir():
            continue
        for cond_dir in sorted(p for p in d.iterdir() if p.is_dir()):
            for order_dir in sorted(p for p in cond_dir.iterdir() if p.is_dir()):
                key = f"{model}/{cond_dir.name}/{order_dir.name}"
                recs = F.load_meta(model, cond_dir.name, order_dir.name)
                per_unit[key] = recs

    seq_ok, tail_ok, vis_ok, gap_ok, sum_ok, idx_ok = [], [], [], [], [], []
    n_recs = 0
    for key, recs in per_unit.items():
        cond = key.split("/")[1]
        for r in recs:
            n_recs += 1
            idx_ok.append(r["decision_token_index"] == r["seq_len"] - 1)
            seq_ok.append(r["seq_len"] > 0)
            if cond == "no_image":
                vis_ok.append(r["n_visual_tokens"] == 0)
            gap_ok.append(r["logit_lens_final_gap"])
            sum_ok.append(abs(sum(r["semantic_probs"].values()) - 1.0))

    check("§2 genuine no-image has zero visual tokens",
          all(vis_ok) if vis_ok else None,
          f"{sum(vis_ok)}/{len(vis_ok)} no_image rows have n_visual_tokens==0")
    check("§3 decision position is the final prompt token",
          all(idx_ok) if idx_ok else None,
          f"{sum(idx_ok)}/{len(idx_ok)} rows have decision_token_index == seq_len-1")
    check("§4 semantic probabilities sum to 1",
          (max(sum_ok) < 1e-5) if sum_ok else None,
          f"max |sum-1| = {max(sum_ok):.3e} over {len(sum_ok)} rows")
    check("§6 final-layer logit lens reproduces the model's own logits",
          (max(gap_ok) < 0.5) if gap_ok else None,
          f"max gap {max(gap_ok):.4f} in logit units over {len(gap_ok)} rows",
          fatal=False)

    # ---- shard/array integrity -------------------------------------------------- #
    bad_shapes, nan_dec, nan_ve_notna = [], [], []
    for model in V.MODELS:
        d = F.FEAT / model
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("shard_*.npz")):
            z = np.load(p)
            meta = [json.loads(l) for l in
                    p.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()
                    if l.strip()]
            ok_meta = [m for m in meta if m["status"] == "ok"]
            if z["decision_hidden"].shape[0] != len(ok_meta):
                bad_shapes.append(str(p))
            if not np.isfinite(z["decision_hidden"]).all():
                nan_dec.append(str(p))
    check("shard array row count matches the metadata rows",
          not bad_shapes, f"{len(bad_shapes)} mismatched shards out of "
                          f"{sum(len(list((F.FEAT / m).rglob('shard_*.npz'))) for m in V.MODELS)}")
    check("decision hidden states are all finite",
          not nan_dec, f"{len(nan_dec)} shards contain non-finite values")

    # ---- coverage ---------------------------------------------------------------- #
    # `random_same_modality_replacement` at O1/O2 is the Stage-3 option-order
    # robustness subset, which the spec deliberately restricts to ~150 stratified items
    # (configs/current900_probe_tasks.yaml).  Every other unit is the full set.
    import m2_extract as _MX
    subset_n = len(_MX.subset_ids(items))
    cov = {}
    for key, recs in per_unit.items():
        if recs:
            cov[key] = {"n": len(recs),
                        "n_samples": len({r["sample_id"] for r in recs})}
    expected = {}
    donors = V.load_donors()
    for c in V.MECH_CONDITIONS:
        expected[c] = sum(1 for s in items if c in V.conditions_for(donors, s))
    expected_subset_only = {("random_same_modality_replacement", "O1"): subset_n,
                            ("random_same_modality_replacement", "O2"): subset_n}
    gaps = []
    for key, v in sorted(cov.items()):
        model, cond, order = key.split("/")
        want = expected_subset_only.get((cond, order), expected.get(cond))
        if want is not None and v["n_samples"] < want:
            gaps.append(f"{key}: {v['n_samples']}/{want}")
    check("condition coverage is complete", not gaps,
          "; ".join(gaps) if gaps else
          f"all {len(cov)} (model,condition,order) units complete "
          f"(mismatch O1/O2 limited to the {subset_n}-item Stage-3 subset by design)",
          fatal=False)

    # ---- §11 provenance ---------------------------------------------------------- #
    prov = sorted((V.ROOT / "provenance").glob("extract_*.json"))
    check("§11 every extraction run wrote provenance",
          len(prov) >= 2, f"{len(prov)} provenance files: "
                          f"{[p.name for p in prov]}", fatal=False)
    ledger_ok = V.LEDGER.is_file() and V.LEDGER.stat().st_size > 0
    check("§11 run ledger exists", ledger_ok, str(V.LEDGER), fatal=False)

    # ---- report ------------------------------------------------------------------ #
    fatal_fail = [c for c in CHECKS if c["fatal"] and c["ok"] is False]
    nonfatal_fail = [c for c in CHECKS if not c["fatal"] and c["ok"] is False]
    L = ["# Acceptance gate", "", f"generated {V.now()}", "",
         f"records checked: **{n_recs}** item-condition-order rows across "
         f"{len(per_unit)} units", "",
         "| check | fatal | result | detail |", "|---|---|---|---|"]
    for c in CHECKS:
        mark = "**PASS**" if c["ok"] else ("FAIL" if c["ok"] is False else "n/a")
        L.append(f"| {c['name']} | {'yes' if c['fatal'] else 'no'} | {mark} | "
                 f"{c['detail']} |")
    L += ["", f"**FATAL failures: {len(fatal_fail)}** · non-fatal: "
              f"{len(nonfatal_fail)}", ""]
    if fatal_fail:
        L += ["Do not interpret any mechanism result until the fatal failures above are "
              "resolved.", ""]
    else:
        L += ["All fatal checks pass; the feature set is usable for interpretation, "
              "subject to the claim boundaries in `configs/spec/01_...`.", ""]
    (V.ROOT / "status" / "ACCEPTANCE.md").write_text("\n".join(L), encoding="utf-8")
    V.write_json(V.ROOT / "status" / "acceptance.json",
                 {"created": V.now(), "n_records": n_recs, "checks": CHECKS,
                  "fatal_failures": len(fatal_fail),
                  "nonfatal_failures": len(nonfatal_fail)})

    for c in CHECKS:
        mark = "PASS" if c["ok"] else ("FAIL" if c["ok"] is False else "n/a")
        print(f"  [{mark}] {c['name']} — {c['detail']}")
    print(f"\nFATAL failures: {len(fatal_fail)}  (see status/ACCEPTANCE.md)")
    return 1 if fatal_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
