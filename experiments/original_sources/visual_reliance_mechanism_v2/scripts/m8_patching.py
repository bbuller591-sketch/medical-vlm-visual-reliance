#!/usr/bin/env python
"""Stage 7 -- activation patching between paired conditions (spec 04 Stage 7, 07 §8).

Evidence level 4 when the identity controls pass: replacing the **decision-position**
residual stream after decoder layer k in a recipient run with the donor run's residual
at the same position and layer, then measuring how far the semantic GT margin moves.

Only within a checkpoint, only at the decision position, only at the same layer.  A
residual at the last prompt position cannot leak into other positions because attention
is causal, so this is a clean interchange intervention on the answer computation.

Pairs (all paired on the SAME source sample):
    clean        -> no_image                          and reverse
    no_image     -> clean                             (the reverse above)
    clean        -> random_same_modality_replacement  and reverse
    clean        -> blur_severe                       and reverse
    opposite_answer_donor <-> clean                   where a valid donor exists

Outcome (primary):      delta_margin = margin_patched - margin_recipient
Outcome (secondary):    recovery     = (margin_patched - margin_recipient) /
                                       (margin_donor   - margin_recipient)
Recovery is a descriptive ratio, clipped and flagged when the denominator is small; it
is NEVER reported as "the fraction of the answer that comes from the image".

Identity controls (spec §8), run before any real patching:
    * hook-disabled forward == baseline (bitwise-equal margins);
    * self-patch (recipient patched with its own activation) -> delta ~ 0;
    * random-sample donor patch -> no systematic target-aligned improvement.

Writes causal/patch_results.json, reports/PATCHING.md, plots/patch_effect_by_layer.png
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np   # noqa: E402
import torch         # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v2 as V      # noqa: E402
import featlib as F        # noqa: E402
import m2_extract as MX    # noqa: E402  (build_inputs / _resolve are reused verbatim)

OUT = V.ROOT / "causal"
COARSE = [3, 7, 11, 15, 19, 23, 27]
PAIRS = [("clean", "no_image"), ("no_image", "clean"),
         ("clean", "random_same_modality_replacement"),
         ("random_same_modality_replacement", "clean"),
         ("clean", "blur_severe"), ("blur_severe", "clean"),
         ("clean", "opposite_answer_donor"),
         ("opposite_answer_donor", "clean")]


DEC_CACHE: dict = {}
ENC_CACHE: dict = {}


def load_condition_decision(model: str, cond: str) -> dict:
    """sample_id -> decision_hidden [T, hidden], loaded once per (model, condition).

    Scanning shards per lookup would have made this stage minutes-per-case instead of
    minutes-per-checkpoint.
    """
    key = (model, cond)
    if key in DEC_CACHE:
        return DEC_CACHE[key]
    out: dict = {}
    d = F.FEAT / model / cond / "O0"
    if d.is_dir():
        for p in sorted(d.glob("shard_*.npz")):
            z = np.load(p)
            ids = [str(x) for x in z["sample_ids"]]
            arr = z["decision_hidden"]
            for i, s in enumerate(ids):
                out[s] = arr[i]
    DEC_CACHE[key] = out
    return out


def donor_vec(model: str, cond: str, sample_id: str, t: int) -> np.ndarray | None:
    """Stored decision state of `cond` for `sample_id` at hidden row t."""
    a = load_condition_decision(model, cond).get(sample_id)
    return None if a is None else a[t].astype(np.float32)


def margin_of(cand_logits: np.ndarray, ordered: dict, n_options: int,
              gt_sem: int) -> tuple[float, float, float]:
    """(GT margin, p(GT), best-wrong p) over the candidates actually on screen.

    Slicing to n_options first matters: for a 2-option item the display letters are
    only A,B, and normalising over A,B,C,D would divide by mass the model was never
    offered.  Matches the extraction instrument exactly.
    """
    z = cand_logits[:n_options].astype(np.float64)
    z = z - z.max()
    p_disp = np.exp(z)
    p_disp /= p_disp.sum()
    p_sem = np.zeros(n_options)
    for disp, sem in enumerate(ordered["permutation"][:n_options]):
        p_sem[sem] = p_disp[disp]
    others = np.delete(p_sem, gt_sem)
    return (float(p_sem[gt_sem] - (others.max() if len(others) else 0.0)),
            float(p_sem[gt_sem]),
            float(others.max() if len(others) else 0.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(V.MODELS))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap cases per phenotype (debug / time control)")
    ap.add_argument("--layers", default="",
                    help="comma-separated hidden-state ROW indices to patch, e.g. "
                         "'16,17,18'. Defaults to the coarse grid.")
    ap.add_argument("--tag", default="",
                    help="suffix for the output filename so a fine scan does not "
                         "overwrite the coarse scan (e.g. --tag fine17)")
    ap.add_argument("--refine", action="store_true",
                    help="after the scan, rescan +-3 rows around the strongest row of "
                         "each pair and merge the results into the same rows")
    args = ap.parse_args()
    scan = ([int(x) for x in args.layers.split(",") if x.strip()]
            if args.layers else list(COARSE))
    scan = sorted(set(scan))

    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    model_label = args.model
    device = f"cuda:{args.gpu}"
    mdir = V.MODELS[model_label]
    run_id = f"current900_patching_{model_label}"

    V.ledger_upsert(run_id, priority=5, dataset="official900", model=model_label,
                    condition="patching", status="running", gpu_id=args.gpu,
                    start_time=V.now(), config_path="configs/current900_probe_tasks.yaml")

    casebook = V.read_json(OUT / "casebook.json")
    items = V.load_items()
    cases = {k: v for k, v in casebook["phenotypes"].items() if v}
    if args.limit:
        cases = {k: v[:args.limit] for k, v in cases.items()}
    n_cases = sum(len(v) for v in cases.values())

    proc = AutoProcessor.from_pretrained(str(mdir))
    proc.image_processor.min_pixels = V.MATCHED_MIN_PIXELS
    proc.image_processor.max_pixels = V.MATCHED_MAX_PIXELS
    image_token_id = proc.tokenizer.convert_tokens_to_ids(proc.image_token)
    cand_ids = {L: proc.tokenizer.encode(L, add_special_tokens=False)[0]
                for L in "ABCDZ"}
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(mdir), torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        low_cpu_mem_usage=True).to(device).eval()
    layers = MX._resolve(model, ["model.language_model.layers", "model.layers",
                                 "language_model.layers"], "decoder layers")[0]
    n_layers = len(layers)
    print(f"{model_label}: {n_layers} decoder layers, {n_cases} cases", flush=True)

    patch_box: dict = {"vec": None}
    cur_layer = {"k": -1}

    def patch_hook(_m, _inp, out):
        if patch_box["vec"] is None:
            return out
        v = torch.as_tensor(patch_box["vec"], dtype=out[0].dtype, device=out[0].device)
        h = out[0].clone()
        h[:, -1, :] = v
        return (h,) + tuple(out[1:])

    def run(enc, ordered, item, k=None, vec=None):
        """One forward, optionally patching hidden-state row k.

        Row k for k>=1 is the output of decoder layer k, i.e. the state ENTERING layer
        k+1.  `layers` is 0-indexed, so hooking `layers[k-1]` is what produces row k.
        Getting this wrong by one would patch the wrong depth and silently mislabel
        every causal result, so the mapping is asserted in the self-patch control.
        """
        handles = []
        if k is not None:
            patch_box["vec"] = vec
            handles.append(layers[k - 1].register_forward_hook(patch_hook))
        try:
            with torch.no_grad():
                o = model(**enc, logits_to_keep=1, use_cache=False)
            lg = o.logits[0, -1].float().cpu().numpy()
            return {L: float(lg[cand_ids[L]]) for L in "ABCDZ"}
        finally:
            for h in handles:
                h.remove()
            patch_box["vec"] = None
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # identity controls first (spec §8)
    # ------------------------------------------------------------------ #
    sid0 = cases[sorted(cases)[0]][0]
    it0 = items[sid0]
    enc0, ord0, _, err0 = MX.build_inputs(model, proc, it0, "clean", "O0", device)
    assert not err0, err0
    base0 = run(enc0, ord0, it0)
    base0b = run(enc0, ord0, it0)
    ctl = {"hook_disabled_repeat_bitwise_equal":
           all(base0[L] == base0b[L] for L in base0)}
    base0c = run(enc0, ord0, it0, k=11, vec=np.zeros(1536, dtype=np.float32))
    ctl["zero_patch_changes_output"] = any(base0c[L] != base0[L] for L in base0)
    # self-patch: recipient's own stored activation -> must be a no-op
    sv = donor_vec(model_label, "clean", sid0, 11)
    if sv is not None:
        self_patch = run(enc0, ord0, it0, k=11, vec=sv)
        ctl["self_patch_max_logit_delta"] = float(max(
            abs(self_patch[L] - base0[L]) for L in base0))
        ctl["self_patch_is_noop"] = ctl["self_patch_max_logit_delta"] < 0.05
    print(f"  identity controls: {ctl}", flush=True)

    # ------------------------------------------------------------------ #
    def cached_inputs(sid, cond):
        """Build (and cache) the processor inputs for one sample and condition."""
        key = (sid, cond)
        if key not in ENC_CACHE:
            ENC_CACHE[key] = MX.build_inputs(model, proc, items[sid], cond, "O0",
                                             device)
        return ENC_CACHE[key]

    def scan_one(sid: str, dcond: str, rcond: str, rows_wanted: list[int]) -> list[dict]:
        """Patch one (case, pair) at every row in `rows_wanted`; return the rows.

        Called twice when --refine is on: once on the coarse grid, once on a fine
        neighborhood.  Results are merged by row index, so a re-scan overwrites the
        same key rather than duplicating it.
        """
        item = items[sid]
        dv = {k: donor_vec(model_label, dcond, sid, k) for k in rows_wanted}
        if any(v is None for v in dv.values()):
            return []
        enc, ordered, _, err = cached_inputs(sid, rcond)
        if err:
            return []
        rec_base = run(enc, ordered, item)
        gt = item["gt_index"]
        m_rec, pgt_rec, _ = margin_of(_vec(rec_base), ordered, item["n_options"], gt)
        don_enc, don_ord, _, derr = cached_inputs(sid, dcond)
        if derr:
            return []
        rec_don = run(don_enc, don_ord, item)
        m_don, pgt_don, _ = margin_of(_vec(rec_don), ordered, item["n_options"], gt)
        row = {"phenotype": phen_of[sid], "sample_id": sid,
               "modality": item["modality"], "donor_condition": dcond,
               "recipient_condition": rcond,
               "margin_recipient": m_rec, "pgt_recipient": pgt_rec,
               "margin_donor_condition": m_don, "pgt_donor_condition": pgt_don,
               "delta_denominator": m_don - m_rec, "by_layer": {}}
        for k in rows_wanted:
            out = run(enc, ordered, item, k=k, vec=dv[k])
            mk, pgtk, _ = margin_of(_vec(out), ordered, item["n_options"], gt)
            denom = m_don - m_rec
            rec = None
            if abs(denom) > 1e-6:
                rec = (mk - m_rec) / denom
                if abs(rec) > 5:
                    rec = float(np.sign(rec) * 5)
            row["by_layer"][str(k)] = {
                "delta_margin": mk - m_rec, "margin_patched": mk,
                "pgt_patched": pgtk, "recovery_clipped": rec}
        return [row]

    phen_of = {s: ph for ph, ss in cases.items() for s in ss}
    results: list[dict] = []
    index: dict[tuple, dict] = {}
    for phen, sids in cases.items():
        for i, sid in enumerate(sids):
            for dcond, rcond in PAIRS:
                got = scan_one(sid, dcond, rcond, scan)
                for r in got:
                    key = (sid, dcond, rcond)
                    if key in index:
                        index[key]["by_layer"].update(r["by_layer"])
                    else:
                        results.append(r)
                        index[key] = r
            if (i + 1) % 5 == 0:
                print(f"  {phen}: {i + 1}/{len(sids)} cases", flush=True)
        gc.collect()

    # ---- optional refine: rescan +-3 around each pair's strongest row ------------- #
    if args.refine and results:
        agg: dict = {}
        for r in results:
            agg.setdefault((r["donor_condition"], r["recipient_condition"]),
                           []).append(r)
        extra: set[int] = set()
        for _, rs in agg.items():
            rows = sorted({int(k) for r in rs for k in r["by_layer"]})
            if not rows:
                continue
            best, bd = rows[0], 0.0
            for k in rows:
                d = float(np.mean([r["by_layer"][str(k)]["delta_margin"] for r in rs
                                   if str(k) in r["by_layer"]]))
                if abs(d) > abs(bd):
                    best, bd = k, d
            extra |= {k for k in range(max(1, best - 3), min(n_layers, best + 4))}
        extra -= set(scan)
        extra = sorted(extra)
        print(f"  refine: rescanning rows {extra}", flush=True)
        for (dcond, rcond), rs in agg.items():
            for r in rs:
                for got in scan_one(r["sample_id"], dcond, rcond, extra):
                    r["by_layer"].update(got["by_layer"])
        print(f"  refine done, rows now "
              f"{sorted({int(k) for r in results for k in r['by_layer']})}", flush=True)

    out = {"created": V.now(), "model": model_label, "n_layers": n_layers,
           "coarse_layers": scan, "refined": bool(args.refine),
           "controls": ctl, "n_rows": len(results), "rows": results}
    stem = f"patch_results_{model_label}{('_' + args.tag) if args.tag else ''}"
    out["tag"] = args.tag
    V.write_json(OUT / f"{stem}.json", out)
    V.ledger_upsert(run_id, status="complete", end_time=V.now(),
                    sample_count_done=len(results),
                    output_path=f"causal/{stem}.json")
    print(f"DONE {model_label}: {len(results)} (case, pair) rows")
    return 0


def _vec(d: dict) -> np.ndarray:
    return np.array([d[L] for L in "ABCDZ"], dtype=np.float64)


if __name__ == "__main__":
    raise SystemExit(main())
