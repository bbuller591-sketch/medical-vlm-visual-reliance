#!/usr/bin/env python
"""Stage 8 -- attention knockout: which route carries the image to the answer?

Spec 02 section F, 04 Stage 8, 07 §9.

Rather than reading attention weights, this *changes the attention graph* and measures
the consequence.  Three interventions, each over a chosen decoder-layer window:

    direct  the decision position may no longer attend to visual tokens
    relay   question/text positions may no longer attend to visual tokens
    both    image -> text and image -> decision are both blocked

Why `relay` matters: blocking only the direct route can *underestimate* visual use,
because earlier question tokens may already have cached visual content and then pass it
to the decision position through text-to-text attention.

Masks are additive and applied to the 4-D attention mask that SDPA consumes; the model
already runs under SDPA, so the attention implementation is unchanged (spec §9 asks for
that comparison only when FlashAttention forces a switch).

Identity controls, run first:
    * empty knockout (no positions masked) reproduces the baseline margin;
    * the mask is applied only to the intended query/key classes (counts recorded);
    * the causal mask is untouched (we only ever OR in extra -inf).

Writes causal/knockout_results_<model>.json, reports/KNOCKOUT.md
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
import m2_extract as MX    # noqa: E402

OUT = V.ROOT / "causal"
BIG = -1e9
TOL_MARGIN = 0.05   # margin units; see identity control rationale


def margin_from_logits(lg: np.ndarray, ordered: dict, n_options: int,
                       gt_sem: int) -> tuple[float, float]:
    z = lg[:n_options] - lg[:n_options].max()
    p = np.exp(z)
    p /= p.sum()
    sem = np.zeros(n_options)
    for disp, s in enumerate(ordered["permutation"][:n_options]):
        sem[s] = p[disp]
    others = np.delete(sem, gt_sem)
    return float(sem[gt_sem] - (others.max() if len(others) else 0.0)), float(sem[gt_sem])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(V.MODELS))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=8, help="cases per phenotype")
    ap.add_argument("--layers", default="0:28", help="decoder-layer window a:b (0-based)")
    ap.add_argument("--tag", default="",
                    help="suffix for the output filename so different knockout "
                         "windows do not overwrite each other (e.g. --tag w0_7)")
    args = ap.parse_args()

    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    model_label = args.model
    device = f"cuda:{args.gpu}"
    mdir = V.MODELS[model_label]
    run_id = f"current900_knockout_{model_label}"
    V.ledger_upsert(run_id, priority=6, dataset="official900", model=model_label,
                    condition="attention_knockout", status="running", gpu_id=args.gpu,
                    start_time=V.now(), config_path="configs/current900_probe_tasks.yaml")

    proc = AutoProcessor.from_pretrained(str(mdir))
    proc.image_processor.min_pixels = V.MATCHED_MIN_PIXELS
    proc.image_processor.max_pixels = V.MATCHED_MAX_PIXELS
    image_token_id = proc.tokenizer.convert_tokens_to_ids(proc.image_token)
    cand_ids = {L: proc.tokenizer.encode(L, add_special_tokens=False)[0] for L in "ABCDZ"}
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(mdir), torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        low_cpu_mem_usage=True).to(device).eval()
    layers = MX._resolve(model, ["model.language_model.layers", "model.layers",
                                 "language_model.layers"], "decoder layers")[0]
    a, b = (int(x) for x in args.layers.split(":"))
    window = list(range(a, min(b, len(layers))))

    ko = {"mode": None, "q": None, "k": None, "base": None}
    stats = {"n_masked_calls": 0, "n_masked_entries": 0, "n_base_built": 0}

    def prehook(_mod, args_, kwargs):
        """Replace the attention mask with an explicit causal mask plus the knockout.

        MEASURED, not assumed: in this transformers version `Qwen2VLAttention` receives
        `attention_mask=None` and handles causality internally, so a hook that only
        edits an incoming 4-D mask would silently do nothing (the first knockout run
        returned exactly 0.0000 change for every arm, which is how this was found).
        The mask is therefore built here, causal part included, and the identity control
        below checks that an empty knockout reproduces the baseline.
        """
        if ko["mode"] is None:
            return None
        hs = kwargs.get("hidden_states")
        if hs is None and args_:
            hs = args_[0]
        if hs is None or hs.dim() != 3:
            return None
        q = hs.shape[1]
        if ko["base"] is None or ko["base"].shape[-1] != q:
            idx = torch.arange(q, device=hs.device)
            base = torch.zeros(1, 1, q, q, dtype=torch.float32, device=hs.device)
            # element [i, j] = query i, key j; mask the FUTURE (j > i).  Getting this
            # comparison backwards masks the past instead, which both breaks the
            # identity control and makes the three knockout arms indistinguishable.
            future = idx.unsqueeze(0) > idx.unsqueeze(1)
            base = base.masked_fill(future.unsqueeze(0).unsqueeze(0), float("-inf"))
            ko["base"] = base
            stats["n_base_built"] += 1
        m = ko["base"].clone()
        qi, ki = ko["q"] or [], ko["k"] or []
        if qi and ki:
            qt = torch.as_tensor(qi, device=hs.device, dtype=torch.long)
            kt = torch.as_tensor(ki, device=hs.device, dtype=torch.long)
            m[0, 0, qt[:, None], kt[None, :]] = float("-inf")
            stats["n_masked_entries"] += int(len(qi) * len(ki))
        kwargs["attention_mask"] = m.to(hs.dtype)
        stats["n_masked_calls"] += 1
        return (args_, kwargs)

    handles = [l.self_attn.register_forward_pre_hook(prehook, with_kwargs=True)
               for l in (layers[i] for i in window)]

    def forward(enc, ordered, item, mode=None, qi=None, ki=None):
        ko["mode"] = mode
        ko["q"], ko["k"] = qi, ki
        try:
            with torch.no_grad():
                o = model(**enc, logits_to_keep=1, use_cache=False)
            lg = o.logits[0, -1].float().cpu().numpy()
            return np.array([lg[cand_ids[L]] for L in "ABCDZ"])
        finally:
            ko["mode"] = None
            del o
            torch.cuda.empty_cache()

    casebook = V.read_json(OUT / "casebook.json")
    items = V.load_items()
    cases = {k: v[:args.limit] for k, v in casebook["phenotypes"].items() if v}

    # ---- identity control: empty knockout must reproduce the baseline ---------- #
    sid0 = cases[sorted(cases)[0]][0]
    it0 = items[sid0]
    enc0, ord0, _, err0 = MX.build_inputs(model, proc, it0, "clean", "O0", device)
    assert not err0, err0
    ids0 = enc0["input_ids"][0].tolist()
    vis0 = [i for i, t in enumerate(ids0) if t == image_token_id]
    txt0 = [i for i in range(len(ids0)) if i not in set(vis0)]
    base0 = forward(enc0, ord0, it0)
    empty0 = forward(enc0, ord0, it0, mode="noop", qi=[], ki=[])
    gt0 = it0["gt_index"]
    m_base0, _ = margin_from_logits(base0, ord0, it0["n_options"], gt0)
    m_empty0, _ = margin_from_logits(empty0, ord0, it0["n_options"], gt0)
    floor = abs(m_empty0 - m_base0)
    # MEASURED noise floor.  Supplying an explicit additive causal mask instead of
    # letting SDPA use `is_causal=True` changes the arithmetic path by exactly one
    # bf16 ULP (0.125 logit at magnitudes ~22-26 here).  The baseline itself is
    # bitwise reproducible, so this floor is the arithmetic cost of the intervention
    # mechanism, not model noise.  Any knockout delta must be compared against it.
    ctl = {"empty_knockout_is_noop": bool(floor <= TOL_MARGIN),
           "identity_margin_floor": float(floor),
           "identity_logit_ulp_note": ("explicit additive causal mask vs SDPA "
                                       "is_causal differs by one bf16 ULP"),
           "tolerance_margin": TOL_MARGIN,
           "n_visual_tokens": len(vis0), "n_text_tokens": len(txt0),
           "seq_len": len(ids0), "knockout_window": [window[0], window[-1]]}
    print(f"  identity control: {ctl}", flush=True)

    results = []
    for phen, sids in cases.items():
        for sid in sids:
            item = items[sid]
            enc, ordered, img, err = MX.build_inputs(model, proc, item, "clean", "O0",
                                                     device)
            if err:
                continue
            ids = enc["input_ids"][0].tolist()
            vis = [i for i, t in enumerate(ids) if t == image_token_id]
            if not vis:
                continue
            last = len(ids) - 1
            txt = [i for i in range(len(ids)) if i not in set(vis) and i != last]
            gt = item["gt_index"]
            base = forward(enc, ordered, item)
            m0, p0 = margin_from_logits(base, ordered, item["n_options"], gt)
            row = {"phenotype": phen, "sample_id": sid, "modality": item["modality"],
                   "n_visual_tokens": len(vis), "seq_len": len(ids),
                   "baseline_margin": m0, "baseline_p_gt": p0}
            for mode, qi in (("direct", [last]), ("relay", txt), ("both", txt + [last])):
                lg = forward(enc, ordered, item, mode=mode, qi=qi, ki=vis)
                mk, pk = margin_from_logits(lg, ordered, item["n_options"], gt)
                row[f"{mode}_margin"] = mk
                row[f"{mode}_p_gt"] = pk
                row[f"{mode}_delta_margin"] = mk - m0
            results.append(row)
        print(f"  {phen}: {len(sids)} cases", flush=True)

    for h in handles:
        h.remove()

    out = {"created": V.now(), "model": model_label, "controls": ctl,
           "mask_stats": stats, "n_rows": len(results), "rows": results,
           "interpretation_floor_margin": ctl.get("identity_margin_floor"),
           "invalid": (not ctl.get("empty_knockout_is_noop", False)),
           "invalid_reason": (None if ctl.get("empty_knockout_is_noop") else
                              "the empty-knockout identity control did not reproduce "
                              "the baseline, so every knockout delta below is "
                              "uninterpretable; do not use these numbers")}
    stem = f"knockout_results_{model_label}{('_' + args.tag) if args.tag else ''}"
    out["tag"] = args.tag
    V.write_json(OUT / f"{stem}.json", out)
    V.ledger_upsert(run_id,
                    status=("complete" if ctl.get("empty_knockout_is_noop")
                            else "partial"),
                    end_time=V.now(), sample_count_done=len(results),
                    output_path=f"causal/{stem}.json",
                    blocker_or_error=("" if ctl.get("empty_knockout_is_noop") else
                                      "empty-knockout identity control failed"))
    if out["invalid"]:
        print(f"WARNING {model_label}: empty-knockout identity control FAILED — "
              f"results marked invalid and must not be interpreted")
    print(f"DONE {model_label}: {len(results)} cases knocked out")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
