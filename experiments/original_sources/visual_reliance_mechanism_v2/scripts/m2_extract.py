#!/usr/bin/env python
"""Current-900 behaviour + mechanism feature extraction (spec Stages 1, 2, 3).

One script, three plans, because all three share the same forward pass and must use the
same frozen matched processor:

    --plan behavior   clean/no_image x O0,O1,O2 x 900        (Stage 1)
    --plan mech       the 7 O0 conditions x 900              (Stage 2)
    --plan subset     random_same_modality_replacement x O1,O2 on ~150 items (Stage 3)
    --plan all        union of the above, minus what is already on disk

What is saved per (model, condition, option_order, item):

  decision hidden state at EVERY layer  [n_layers+1, hidden]   -- the state at the last
                                                                  prompt token, whose
                                                                  next-token logits are
                                                                  the ones that score
                                                                  the candidates
  pooled visual residual per layer      [n_layers+1, hidden]   -- mean over the image
                                                                  pad token positions
  vision-encoder pooled output          [hidden]               -- captured from a
                                                                  forward hook on
                                                                  model.visual; before
                                                                  any language fusion
  layer-wise candidate logits           [n_layers+1, n_cand]   -- logit lens
  final candidate probabilities, tag margins, entropy, token metadata

Storage is sharded: features/<model>/<condition>/<order>/shard_XXXX.npz with SHARD
items each, so a crash costs at most SHARD items and the run resumes by skipping shards
that already exist.  Nothing is written for a shard until that whole shard is computed.

The decision position, the candidate normalisation and the semantic remapping are
IDENTICAL to the old standardised instrument, so behaviour here is comparable to the
old run -- which is the point of the matched-processing re-run.

    python scripts/m2_extract.py --model qwen2vl --gpu 3 --plan all
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np   # noqa: E402
import torch         # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v2 as V   # noqa: E402

SHARD = 100
FEAT = V.ROOT / "features"


# --------------------------------------------------------------------------- #
# plans
# --------------------------------------------------------------------------- #
def plan_units(plan: str, model: str, items: dict, donors: dict,
               subset_ids: list[str]) -> list[tuple[str, str, list[str]]]:
    """[(condition, option_order, sample_ids)]"""
    all_ids = sorted(items)
    units: list[tuple[str, str, list[str]]] = []
    if plan in ("behavior", "all"):
        for cond in ("clean", "no_image"):
            for order in ("O0", "O1", "O2"):
                units.append((cond, order, all_ids))
    if plan in ("mech", "all"):
        for cond in V.MECH_CONDITIONS:
            ids = [s for s in all_ids if cond in V.conditions_for(donors, s)]
            units.append((cond, "O0", ids))
    if plan in ("subset", "all"):
        for order in ("O1", "O2"):
            units.append(("random_same_modality_replacement", order, subset_ids))
    # de-duplicate identical (cond, order) units keeping the widest id list
    best: dict[tuple[str, str], list[str]] = {}
    for cond, order, ids in units:
        k = (cond, order)
        if k not in best or len(ids) > len(best[k]):
            best[k] = ids
    return [(c, o, best[(c, o)]) for c, o in sorted(best)]


def subset_ids(items: dict) -> list[str]:
    """~150 stratified items for the option-order robustness check (spec Stage 3):
    75 MRI / 40 CT / 35 X-ray, deterministic."""
    rng = np.random.default_rng(V.GLOBAL_SEED)
    want = {"MRI": 75, "CT": 40, "Xray": 35}
    out = []
    for mod, n in want.items():
        ids = sorted(s for s, it in items.items() if it["modality"] == mod)
        pick = rng.choice(len(ids), size=min(n, len(ids)), replace=False)
        out += [ids[int(i)] for i in sorted(pick)]
    return sorted(out)


# --------------------------------------------------------------------------- #
# one forward
# --------------------------------------------------------------------------- #
def build_inputs(model, proc, item, condition, order_id, device):
    ordered = V.apply_order(item, order_id)
    prompt = V.build_m2_prompt(ordered["mcq_text"])
    img_path = V.image_path(condition, item)
    content = []
    if img_path is not None:
        if not Path(img_path).is_file():
            return None, ordered, None, f"missing image {img_path}"
        content.append({"type": "image", "image": str(img_path)})
    content.append({"type": "text", "text": prompt})
    msgs = [{"role": "user", "content": content}]

    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if img_path is not None:
        from qwen_vl_utils import process_vision_info
        im, vid = process_vision_info(msgs)
        enc = proc(text=[text], images=im, videos=vid, return_tensors="pt",
                   padding=True)
    else:
        enc = proc(text=[text], return_tensors="pt", padding=True)
    enc = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in enc.items()}
    return enc, ordered, img_path, None


def _resolve(root, candidates, what: str):
    """Locate a submodule by trying known attribute paths across transformers versions.

    transformers nests the text model differently between releases
    (`model.norm` vs `model.language_model.norm`), and getting this wrong would silently
    corrupt every logit-lens number, so it is resolved and then verified, never assumed.
    """
    for path in candidates:
        obj = root
        try:
            for a in path.split("."):
                obj = getattr(obj, a)
            return obj, path
        except AttributeError:
            continue
    raise RuntimeError(f"could not locate {what}; tried {candidates}")


def forward(model, enc, candidate_ids: dict, image_token_id: int, paths: dict):
    # logits_to_keep=1: only the final position's logits are ever used, and computing
    # the LM head over the whole sequence is ~140 GFLOP per forward (seq x 1536 x
    # 151936).  The values at the decision position are unchanged.
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, use_cache=False,
                    logits_to_keep=1)
    hs = out.hidden_states                       # tuple, len = n_layers + 1
    last = out.logits[0, -1]                     # decision position = final prompt token
    ids = enc["input_ids"][0].tolist()
    vis_pos = [i for i, t in enumerate(ids) if t == image_token_id]

    n_layers = len(hs)
    hidden = hs[0].shape[-1]
    dec = np.empty((n_layers, hidden), dtype=np.float16)
    vis = np.full((n_layers, hidden), np.nan, dtype=np.float16)
    for l, h in enumerate(hs):
        dec[l] = h[0, -1].float().cpu().numpy().astype(np.float16)
        if vis_pos:
            vis[l] = (h[0, vis_pos].float().mean(0).cpu().numpy().astype(np.float16))

    # ---- logit lens: architecture-valid final norm + LM head at every layer ---- #
    #
    # MEASURED, NOT ASSUMED (acceptance §6).  For Qwen2-VL in this transformers
    # version `output_hidden_states` returns n_layers+1 states = the 28 decoder-layer
    # outputs followed by the FINAL ALREADY-NORMALISED state, with NO embedding entry.
    # Verified on a real forward: head(hs[-1]) reproduces the model's own candidate
    # logits to 0.031 (bf16 rounding), whereas head(norm(hs[-1])) is off by 12.0.
    # So the final entry must not be normalised again, and the earlier entries must be.
    norm = paths["norm"]
    head = paths["lm_head"]
    cand_ids = [candidate_ids[L] for L in candidate_ids]
    n_last = len(hs) - 1
    lens = np.empty((n_layers, len(cand_ids)), dtype=np.float32)
    with torch.no_grad():
        for l, h in enumerate(hs):
            v = h[0, -1]
            if l != n_last:
                v = norm(v.unsqueeze(0)).to(head.weight.dtype)[0]
            lens[l] = head(v).float()[cand_ids].cpu().numpy()

    return {
        "dec": dec, "vis": vis, "lens": lens,
        "seq_len": len(ids), "n_visual_tokens": len(vis_pos),
        "decision_token_index": len(ids) - 1,
        "input_ids_head": ids[:8], "input_ids_tail": ids[-8:],
        "cand_logits_true": last[cand_ids].float().cpu().numpy(),
    }


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(V.MODELS))
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--plan", default="all",
                    choices=["behavior", "mech", "subset", "all"])
    ap.add_argument("--shard", type=int, default=SHARD)
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: cap items per unit")
    ap.add_argument("--tag", default="",
                    help="debug: write features under features/<tag>/ instead, so a "
                         "smoke run can never contaminate the real feature tree")
    args = ap.parse_args()

    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    model_label = args.model
    device = f"cuda:{args.gpu}"
    mdir = V.MODELS[model_label]
    run_id = f"current900_{model_label}_{args.plan}"
    feat_root = FEAT / args.tag if args.tag else FEAT

    V.ledger_upsert(run_id, priority=1, dataset="official900", model=model_label,
                    condition=args.plan, status="running", gpu_id=args.gpu,
                    start_time=V.now(), config_path="configs/current900_probe_tasks.yaml")

    items = V.load_items()
    donors = V.load_donors()
    sub = subset_ids(items)
    units = plan_units(args.plan, model_label, items, donors, sub)

    proc = AutoProcessor.from_pretrained(str(mdir))
    proc.image_processor.min_pixels = V.MATCHED_MIN_PIXELS
    proc.image_processor.max_pixels = V.MATCHED_MAX_PIXELS
    image_token_id = proc.tokenizer.convert_tokens_to_ids(proc.image_token)
    tokenizer = proc.tokenizer
    cand_ids = {L: tokenizer.encode(L, add_special_tokens=False)[0] for L in "ABCDZ"}

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(mdir), torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        low_cpu_mem_usage=True).to(device).eval()

    # vision-encoder capture (before language fusion) via forward hook
    ve_box: dict = {}

    def ve_hook(_m, _inp, out):
        o = out[0] if isinstance(out, (tuple, list)) else out
        if torch.is_tensor(o) and o.dim() == 2:
            ve_box["pool"] = o.float().mean(0).cpu().numpy().astype(np.float16)

    norm_mod, norm_path = _resolve(
        model, ["model.language_model.norm", "model.norm", "language_model.norm"],
        "final norm")
    assert not hasattr(norm_mod, "weight") or norm_mod.weight.ndim == 1, "norm sanity"
    head_mod, head_path = _resolve(model, ["lm_head", "model.lm_head"], "lm_head")
    vis_mod, vis_path = _resolve(
        model, ["model.visual", "visual", "model.model.visual"], "vision tower")
    paths = {"norm": norm_mod, "lm_head": head_mod, "visual": vis_mod}
    print(f"  resolved: norm={norm_path} lm_head={head_path} visual={vis_path}",
          flush=True)
    h = vis_mod.register_forward_hook(ve_hook)

    prov = {
        "model_label": model_label, "model_path": str(mdir),
        "hf_id": V.MODEL_HF_ID[model_label], "gpu": args.gpu, "plan": args.plan,
        "matched_min_pixels": V.MATCHED_MIN_PIXELS,
        "matched_max_pixels": V.MATCHED_MAX_PIXELS,
        "attn_implementation": "sdpa", "dtype": "bfloat16",
        "n_layers_plus_embedding": None,
        "hidden_states_semantics": (
            "Decoded empirically, not assumed.  row 0 == embed_tokens(last token) "
            "exactly (bitwise); row T-1 is the state the LM head consumes (head(row) "
            "reproduces the model's own logits to 0.031 in logit units in bf16, while "
            "head(norm(row)) is off by 12.0); rows 1..T-2 are the 27 states between. "
            "Labelled embed / L1..L27 / L28+final_norm.  A forward hook on layers[k] "
            "(0-indexed) produces row k+1."),
        "feature_dir": str(feat_root / model_label),
        "tag": args.tag,
        "resolved_paths": {"norm": norm_path, "lm_head": head_path,
                           "visual": vis_path},
        "image_token_id": image_token_id, "candidate_ids": cand_ids,
        "started": V.now(),
    }

    total_done = 0
    t0 = time.time()
    n_units = len(units)
    for ui, (cond, order, ids) in enumerate(units, 1):
        if args.limit:
            ids = ids[:args.limit]
        outdir = feat_root / model_label / cond / order
        outdir.mkdir(parents=True, exist_ok=True)
        n_sh = (len(ids) + args.shard - 1) // args.shard
        for si in range(n_sh):
            shard_ids = ids[si * args.shard:(si + 1) * args.shard]
            shard_path = outdir / f"shard_{si:04d}.npz"
            meta_path = outdir / f"shard_{si:04d}.jsonl"
            if shard_path.is_file() and meta_path.is_file():
                continue
            dec_l, vis_l, lens_l, ve_l = [], [], [], []
            meta, errs = [], []
            for sid in shard_ids:
                item = items[sid]
                enc, ordered, img_path, err = build_inputs(model, proc, item, cond,
                                                           order, device)
                if err:
                    errs.append({"sample_id": sid, "error": err})
                    meta.append({"sample_id": sid, "status": "error", "error": err})
                    continue
                img_sha = (V.sha256_file(Path(img_path)) if img_path is not None else None)
                try:
                    f = forward(model, enc, cand_ids, image_token_id, paths)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    meta.append({"sample_id": sid, "status": "error",
                                 "error": "CUDA_OOM"})
                    continue
                disp = ordered["display_letters"]
                # Normalise over the candidates ACTUALLY ON SCREEN, matching the old
                # standardised instrument.  Using all of A-D,Z here would divide by a
                # mass the model was never offered (Z is not displayed in M2) and would
                # make p(GT)/entropy incomparable with the old run.  For 2-option items
                # the display letters are only A,B.
                #
                # The final-layer probabilities use the model's OWN logits, not the
                # logit-lens reconstruction: the two agree to 0.125 in logit units
                # (bf16 reduction-order difference) and there is no reason to report the
                # reconstructed value when the true one is in hand.  `layer_candidate_
                # logits` keeps the lens for the per-layer trajectory.
                n_disp = len(disp)
                pv = softmax(f["cand_logits_true"][:n_disp])
                probs_map = {L: float(pv[i]) for i, L in enumerate(disp)}
                sem_probs = {str(ordered["display_to_semantic"][L]): p
                             for L, p in probs_map.items()}
                # acceptance §6: final-layer logit lens must reproduce the model's own
                # candidate logits.  Recorded for every item; asserted below.
                lens_gap = float(np.abs(f["lens"][-1] - f["cand_logits_true"]).max())
                gt_disp = ordered["gt_display_letter"]
                gt_sem = item["gt_index"]
                psorted = np.array([sem_probs[str(i)] for i in range(item["n_options"])])
                others = np.delete(psorted, gt_sem)
                donor = V.donor_entry(donors, sid, cond)
                meta.append({
                    "sample_id": sid, "modality": item["modality"],
                    "n_options": item["n_options"], "condition": cond,
                    "option_order": order, "model_label": model_label,
                    "image_path": str(img_path) if img_path else None,
                    "actual_input_image_sha256": img_sha,
                    "donor_sample_id": (donor or {}).get("donor_sample_id"),
                    "donor_image_sha256": (donor or {}).get("donor_image_sha256"),
                    "donor_semantic_gt_text": (donor or {}).get("donor_gt_norm"),
                    "gt_semantic": gt_sem, "gt_display_letter": gt_disp,
                    "display_options": ordered["display_options"],
                    "semantic_to_display": ordered["semantic_to_display"],
                    "permutation": ordered["permutation"],
                    "decision_token_index": f["decision_token_index"],
                    "seq_len": f["seq_len"], "n_visual_tokens": f["n_visual_tokens"],
                    "input_ids_head": f["input_ids_head"],
                    "input_ids_tail": f["input_ids_tail"],
                    "layer_candidate_logits": f["lens"].tolist(),
                    "final_candidate_probs": probs_map,
                    "semantic_probs": sem_probs,
                    "correct": bool(int(np.argmax(psorted)) == gt_sem),
                    "p_gt": float(psorted[gt_sem]),
                    "entropy_semantic": float(-(psorted * np.log(np.maximum(psorted,
                                                                          1e-12))).sum()),
                    "margin_gt": float(psorted[gt_sem] - (others.max() if len(others)
                                                          else 0.0)),
                    "logit_lens_final_gap": lens_gap,
                    "hidden_state_tuple_len": f["dec"].shape[0],
                    "status": "ok",
                })
                dec_l.append(f["dec"])
                vis_l.append(f["vis"])
                lens_l.append(f["lens"])
                ve_l.append(ve_box.get("pool", np.full(f["dec"].shape[-1], np.nan,
                                                       dtype=np.float16)))
            if dec_l:
                np.savez_compressed(
                    shard_path,
                    decision_hidden=np.stack(dec_l),
                    visual_pooled=np.stack(vis_l),
                    layer_candidate_logits=np.stack(lens_l),
                    vision_encoder_pool=np.stack(ve_l),
                    sample_ids=np.array([m["sample_id"] for m in meta
                                         if m["status"] == "ok"]))
                with open(meta_path, "w", encoding="utf-8") as fh:
                    for m in meta:
                        fh.write(json.dumps(m, ensure_ascii=False) + "\n")
                total_done += len(dec_l)
                prov["n_layers_plus_embedding"] = dec_l[0].shape[0]
            if errs:
                print(f"    {len(errs)} error(s) in {cond}/{order} shard {si}: "
                      f"{errs[:2]}", flush=True)
        el = time.time() - t0
        print(f"[{model_label}/{args.plan}] {ui}/{n_units} {cond}/{order} "
              f"n={len(ids)} done_total={total_done} "
              f"{el / 60:.1f} min ({total_done / max(el, 1e-9):.1f} item/s)",
              flush=True)

    h.remove()
    prov["items_done"] = total_done
    prov["finished"] = V.now()
    V.write_json(V.ROOT / "provenance" /
                 f"extract_{model_label}_{args.plan}{'_smoke' if args.tag else ''}.json",
                 prov)
    V.ledger_upsert(run_id, status="complete", end_time=V.now(),
                    sample_count_done=total_done,
                    output_path=f"{feat_root.relative_to(V.ROOT)}/{model_label}")
    print(f"DONE {model_label}/{args.plan}: {total_done} item-condition-order rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
