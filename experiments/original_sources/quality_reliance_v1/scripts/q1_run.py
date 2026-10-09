#!/usr/bin/env python
"""Q1 -- Quality x Relevance behaviour + decision-state extraction (shardable).

    python scripts/q1_run.py --model qwen2vl --arms all --orders O0,O1,O2 \
        --shard 0 --nshards 4 --device cuda:1 --worker w0

WHAT IS MEASURED PER (item, arm, order)
---------------------------------------
One teacher-forced forward pass over the frozen MCQ prompt gives
  * the log-probability of each candidate letter at the decision position, hence
    the normalised A/B/C/D distribution and everything derived from it
    (correct margin, entropy, JS to no-image);
  * optionally the hidden state at that same decision position for every row, so
    Level-1 probes never need a second pass over the model.

The decision position is the last position of the prefill -- i.e. the position at
which the model chooses the answer.  This is the same position the old instrument
used, which is what makes the two runs comparable.

PROCESSING
----------
min_pixels/max_pixels are forced to the MATCHED values for both checkpoints
(see common_qr's docstring §2).  Without this, a 4240x3480 image is worth ~30x
more visual tokens to Qwen2-VL-2B than to MedVLM-R1, and every cross-model
statement about "how much the model used the image" would be confounded by how
many tokens it was handed.

RESUME
------
Every record carries `resume_key` over (model, arm, order, sample, image sha256,
processor id, prompt hash).  A re-run skips keys already present in the shard
files, so a killed job costs at most the in-flight item.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_qr as Q   # noqa: E402

Q.cuda_alloc_env()


# --------------------------------------------------------------------------- #
def parse_arms(spec: str) -> list[str]:
    if spec in ("all", "image", "matched"):
        return list(Q.IMAGE_ARMS)
    if spec == "core":
        # the essential Stage-1 set: all four matched levels, all three donors at
        # clean and severe, plus no_image.  Intermediate severities follow.
        out = [Q.MATCHED_ARM[q] for q in Q.QUALITY_LEVELS]
        out += [Q.mismatch_arm(q, k) for q in ("clean", "severe")
                for k in range(1, Q.N_DONORS + 1)]
        return out
    return [a for a in spec.split(",") if a]


class Runner:
    def __init__(self, model_label: str, device: str, attn: str, max_pixels: int,
                 soft_limit_mib: int = 0):
        import torch
        from transformers import AutoProcessor

        self.torch = torch
        self.label = model_label
        self.device = device
        if soft_limit_mib > 0:
            # A hard ceiling, not advice.  This process is a guest on a GPU that
            # somebody else is paying for; capping its own allocator is the only
            # way to guarantee it can never be the reason that job dies.
            idx = int(str(device).split(":")[-1])
            total = torch.cuda.get_device_properties(idx).total_memory
            frac = min(0.95, soft_limit_mib * 1024 * 1024 / total)
            torch.cuda.set_per_process_memory_fraction(frac, idx)
            self.soft_limit_frac = frac
        else:
            self.soft_limit_frac = None
        path = Q.MODELS[model_label]
        self.model = self._load(path, attn, device)
        self.processor = AutoProcessor.from_pretrained(path)
        self._force_processor(max_pixels)
        self.tok = self.processor.tokenizer
        self.hidden_size = self.model.config.text_config.hidden_size
        self.n_rows = self.model.config.text_config.num_hidden_layers + 1
        ip = self.processor.image_processor
        self.proc_info = {"class": type(ip).__name__,
                          "min_pixels": getattr(ip, "min_pixels", None),
                          "max_pixels": getattr(ip, "max_pixels", None),
                          "merge_size": getattr(ip, "merge_size", None),
                          "processor_id": Q.MATCHED_PROCESSOR_ID}

    def _load(self, path, attn, device):
        from transformers import AutoConfig
        import transformers
        cls_name = AutoConfig.from_pretrained(path).architectures[0]
        cls = getattr(transformers, cls_name)
        return cls.from_pretrained(path, torch_dtype="bfloat16",
                                   attn_implementation=attn,
                                   device_map=None).to(device).eval()

    def _force_processor(self, max_pixels: int) -> None:
        """Pin the matched pixel budget on the processor object itself.

        Both attribute styles are set because Qwen2VL and Qwen2_5_VL read the
        budget from different places (`max_pixels` vs a `size` dict), and a silent
        mismatch here would only show up as an odd token count much later.
        """
        ip = self.processor.image_processor
        ip.min_pixels = Q.MATCHED_MIN_PIXELS
        ip.max_pixels = max_pixels
        if hasattr(ip, "size") and isinstance(getattr(ip, "size"), dict):
            ip.size["longest_edge"] = max_pixels
            ip.size["shortest_edge"] = Q.MATCHED_MIN_PIXELS

    # ------------------------------------------------------------------ #
    def build_inputs(self, prompt_text: str, image_path: Path | None):
        """image_path=None is a GENUINE no-image pass: no placeholder token, no
        pixel_values tensor at all."""
        if image_path is None:
            msg = [{"role": "user",
                    "content": [{"type": "text", "text": prompt_text}]}]
            text = self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True)
            return self.processor(text=[text], padding=True,
                                  return_tensors="pt").to(self.device), text
        from qwen_vl_utils import process_vision_info
        msg = [{"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": prompt_text}]}]
        text = self.processor.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True)
        ii, vi = process_vision_info([msg])
        enc = self.processor(text=[text], images=ii, videos=vi, padding=True,
                             return_tensors="pt").to(self.device)
        return enc, text

    def n_visual_tokens(self, enc) -> int:
        if "image_grid_thw" not in enc or enc["image_grid_thw"] is None:
            return 0
        m = self.proc_info["merge_size"] or 1
        return int((enc["image_grid_thw"].prod(-1) // (m ** 2)).sum())

    def forward(self, enc, want_hidden: bool):
        vis_pooled = None
        box: dict = {}
        handle = None
        if want_hidden and hasattr(self.model, "visual"):
            # A pure-vision representation, captured BEFORE the language model
            # sees anything.  The balanced-multiset control needs it: a probe
            # given only this must not be able to tell the matched arm from the
            # mismatch arm when the two classes share one image multiset.  If it
            # can, the construction is broken, not the model.
            def _hook(_m, _i, out):
                box["out"] = out[0] if isinstance(out, (tuple, list)) else out

            handle = self.model.visual.register_forward_hook(_hook)
        try:
            with self.torch.inference_mode():
                out = self.model(**enc, output_hidden_states=want_hidden,
                                 use_cache=False)
        finally:
            if handle is not None:
                handle.remove()
        logits = out.logits[0, -1].float()
        hs = None
        if want_hidden:
            # row 0 is the input embedding, rows 1..L are the layer outputs, and
            # the last row is what the LM head reads.  Stored for the decision
            # position only: full-token dumps for 900 items would be ~100x larger.
            hs = self.torch.stack([h[0, -1].float() for h in out.hidden_states], 0)
            # numpy has no bfloat16 dtype, so the cast to float32 happens on the
            # device and the narrowing to float16 happens at save time
            hs = hs.cpu().numpy()
            v = box.get("out")
            if v is not None and v.numel():
                vis_pooled = v.float().mean(dim=0).cpu().numpy()
        return logits, hs, vis_pooled

    def score(self, logits, letters: list[str]) -> dict:
        lp = self.torch.log_softmax(logits, dim=-1)
        out, modes = {}, {}
        for L in letters:
            ids = self.tok.encode(L, add_special_tokens=False)
            if len(ids) == 1:
                out[L] = float(lp[ids[0]])
                modes[L] = "single_token"
            else:                                     # pragma: no cover
                out[L] = float(sum(float(lp[t]) for t in ids))
                modes[L] = "sum_of_tokens"
        return out, modes


# --------------------------------------------------------------------------- #
def base_record(item, arm, order_id, model_label, revision, proc_info,
                worker_id, gpu_id, prompt_type) -> dict:
    qlevel, rel = Q.ARM_SPEC[arm]
    return {
        "sample_id": item["sample_id"],
        "model_id": Q.MODEL_DISPLAY[model_label],
        "model_label": model_label,
        "model_revision": revision,
        "modality": item["modality"],
        "question_id": item["file_index"],
        "arm": arm,
        "quality_level": qlevel,
        "relevance_type": rel,
        "option_order_id": order_id,
        "prompt_type": prompt_type,
        "instrument_version": "qr-v1",
        "processor_id": proc_info["processor_id"],
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "timestamp": Q.now(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(Q.MODELS))
    ap.add_argument("--arms", default="core")
    ap.add_argument("--orders", default="O0,O1,O2")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--worker", default=None)
    ap.add_argument("--limit", type=int, default=0, help="first N items (smoke)")
    ap.add_argument("--features", action="store_true",
                    help="deprecated no-op: decision states are always stored for "
                         "--feature-orders, because output_hidden_states costs "
                         "almost nothing on a pass that is already being run")
    ap.add_argument("--no-features", action="store_true",
                    help="skip decision-state storage entirely")
    ap.add_argument("--feature-orders", default="O0")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--max-pixels", type=int, default=Q.MATCHED_MAX_PIXELS)
    ap.add_argument("--gpu-soft-limit-mib", type=int, default=0,
                    help="hard cap on this process's GPU memory (guest policy)")
    ap.add_argument("--samples", default="", help="comma list of sample ids")
    args = ap.parse_args()

    arms = parse_arms(args.arms)
    if "no_image" not in arms and args.arms != "all":
        pass                    # caller decides; no_image is usually its own job
    orders = [o for o in args.orders.split(",") if o]
    # Decision states are collected on whichever orders are named here, for every
    # arm, as part of the forward pass that is already being run.  Storing them is
    # not an extra model call, so there is no reason to gate it behind a flag --
    # a job that produced behaviour without states would only have to be re-run.
    feat_orders = set() if args.no_features else set(
        x for x in args.feature_orders.split(",") if x)
    items = Q.load_items()
    ids = sorted(items)
    if args.samples:
        want = set(x for x in args.samples.split(",") if x)
        ids = [i for i in ids if i in want]
    if args.limit:
        ids = ids[:args.limit]
    ids = ids[args.shard::args.nshards]

    worker = args.worker or f"{args.model}_s{args.shard}"
    tag = f"{args.model}__{args.worker}"
    out_dir = Q.ROOT / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"QR__{tag}.jsonl"
    feat_dir = Q.ROOT / "features" / args.model
    feat_dir.mkdir(parents=True, exist_ok=True)

    done = set()
    if raw_path.is_file():
        for r in Q.load_jsonl(raw_path):
            done.add(r["resume_key"])
    print(f"[{worker}] {len(done)} records already present, "
          f"{len(ids)} items x {len(arms)} arms x {len(orders)} orders "
          f"= {len(ids)*len(arms)*len(orders)} planned", flush=True)

    runner = Runner(args.model, args.device, args.attn, args.max_pixels,
                    args.gpu_soft_limit_mib)
    if runner.soft_limit_frac:
        print(f"[{worker}] own GPU budget: {runner.soft_limit_frac:.3f} of device",
              flush=True)
    revision = str(Q.MODELS[args.model])
    prompt_type = "standardized_answer_only"

    # features are accumulated per (arm, order) and flushed once per arm, so a
    # crash never leaves a half-written .npz that looks complete
    import numpy as np
    # Feature files are keyed by (arm, order, WORKER).  Shards partition the items,
    # so a file keyed only by (arm, order) would be written by every shard at once
    # and each would leave only its own 225 items -- and the concurrent writes
    # corrupt the zip container.  The probes merge across workers instead.
    feat_buf: dict[tuple, tuple[list, list, list]] = {}
    for (a, o) in [(a, o) for a in arms for o in orders if o in feat_orders]:
        p = feat_dir / f"{a}__{o}__{worker}.npz"
        if p.is_file():
            continue
        feat_buf[(a, o)] = ([], [], [])

    n_done = 0
    t0 = time.time()
    # The vision tower's projected output width is discovered from the first image
    # actually seen rather than read from the config, because the merger's output
    # width is what comes back and it is not always the config's vision hidden
    # size.  Rows whose width is not yet known are padded with NaN.
    vstate = {"dim": None, "flushed": set()}

    def flush(key) -> None:
        """Write one (arm, order) feature file.  Called per arm and from the
        SIGTERM handler, so a yield never costs more than the current item."""
        if key not in feat_buf or not feat_buf[key][1]:
            return
        sids, arrs, viss = feat_buf[key]
        vis_dim = vstate["dim"] or 0
        vis_arr = np.full((len(viss), vis_dim), np.nan, dtype="float32")
        for i, v in enumerate(viss):
            if v is not None:               # no_image has no vision tokens at all
                vis_arr[i, :v.shape[0]] = v
        import tempfile
        tmp = feat_dir / f"{key[0]}__{key[1]}__{worker}.tmp.npz"
        np.savez_compressed(tmp, sample_ids=np.array(sids),
                            hidden=np.stack(arrs).astype("float16"),
                            vis_pooled=vis_arr.astype("float16"))
        # atomic: a crash mid-write must not leave a file that looks complete
        tmp.replace(feat_dir / f"{key[0]}__{key[1]}__{worker}.npz")
        print(f"[{worker}] features {key[0]}__{key[1]}__{worker}: {len(sids)} x "
              f"{arrs[0].shape[0]} x {arrs[0].shape[1]} + vis x {vis_dim}",
              flush=True)
        vstate["flushed"].add(key)
        feat_buf[key] = ([], [], [])

    def _on_term(_sig, _frm):
        print(f"[{worker}] SIGTERM -- flushing {len(feat_buf)} feature buffers",
              flush=True)
        for k in list(feat_buf):
            flush(k)
        sys.exit(143)

    import signal as _signal
    _signal.signal(_signal.SIGTERM, _on_term)
    for arm in arms:
        for order_id in orders:
            for sid in ids:
                item = items[sid]
                ro = Q.apply_order(item, order_id)
                letters = ro["display_letters"]
                d2s = ro["display_to_semantic"]
                prompt = Q.build_m2_prompt(ro["mcq_text"])
                ipath = Q.image_path(arm, item)
                if ipath is None and arm != Q.NOIMAGE_ARM:
                    continue            # donor missing: skip, never guess
                if ipath is not None and not Path(ipath).is_file():
                    Q.append_jsonl(raw_path, {
                        **base_record(item, arm, order_id, args.model, revision,
                                      runner.proc_info, worker, args.device,
                                      prompt_type),
                        "status": "missing_image", "image_path": str(ipath),
                        "resume_key": f"missing|{arm}|{order_id}|{sid}"})
                    continue
                want_hidden = (arm, order_id) in feat_buf
                rk = Q.stable_seed(args.model, arm, order_id, sid,
                                   "" if ipath is None else Q.sha256_file(ipath),
                                   runner.proc_info["processor_id"],
                                   Q.prompt_hash(arm))
                already = rk in done
                # A shard that is stopped mid-arm (yielding to another user, or
                # OOM) loses whatever was still in the feature buffer.  Those
                # items are already in `done`, so a naive resume would skip them
                # and the feature file would come out short.  When the feature
                # file is missing we therefore recompute the item and write only
                # the states, never a second behaviour row.
                if already and not want_hidden:
                    continue
                try:
                    enc, text = runner.build_inputs(prompt, ipath)
                    logits, hs, vis = runner.forward(enc, want_hidden)
                    logp, modes = runner.score(logits, letters)
                    probs_disp = Q.normalise_probs(logp)
                    # display letter -> semantic index, so that every downstream
                    # comparison is on the semantic option, not on a position
                    sem = {str(d2s[L]): probs_disp[L] for L in letters}
                    sem_log = {str(d2s[L]): logp[L] for L in letters}
                    gt_sem = str(item["gt_index"])
                    gt_letter = ro["gt_display_letter"]
                    pred_letter = max(probs_disp, key=probs_disp.get)
                    pred_sem = d2s[pred_letter]
                    rec = {
                        **base_record(item, arm, order_id, args.model, revision,
                                      runner.proc_info, worker, args.device,
                                      prompt_type),
                        "status": "ok",
                        "option_mapping": {L: d2s[L] for L in letters},
                        "display_options": ro["display_options"],
                        "permutation": ro["permutation"],
                        "gt_semantic": int(gt_sem),
                        "gt_display_letter": gt_letter,
                        "gt_semantic_text": item["gt_text"],
                        "candidate_logprobs": logp,
                        "candidate_probs": probs_disp,
                        "semantic_probs": sem,
                        "semantic_logprobs": sem_log,
                        "pred_display_letter": pred_letter,
                        "pred_semantic": int(pred_sem),
                        "correct": bool(int(pred_sem) == int(gt_sem)),
                        "p_gt_semantic": sem.get(gt_sem),
                        # spec 04 defines M as a LOG margin:
                        #   M = log P(y*) - max_{y!=y*} log P(y)
                        # `gt_margin` (the probability margin the old instrument
                        # used) is kept alongside so the two runs stay comparable.
                        "gt_log_margin": Q.gt_margin(sem_log, gt_sem),
                        "gt_margin": Q.gt_margin(sem, gt_sem),
                        "entropy": Q.entropy(sem),
                        "n_visual_tokens": runner.n_visual_tokens(enc),
                        "image_path": str(ipath) if ipath else None,
                        "image_sha256": (Q.sha256_file(ipath)
                                         if ipath else None),
                        "scoring_mode": modes,
                        "prompt_hash": Q.sha256_text(text),
                        "n_semantic_options": len(letters),
                        "resume_key": rk,
                    }
                    if arm.startswith("pm_"):
                        rec["donor"] = Q.donor_record(sid, int(arm.split("_")[-1][1:]))
                    elif arm.startswith("sc_"):
                        rec["donor"] = Q.semantic_record(sid)
                    if not already:
                        Q.append_jsonl(raw_path, rec)
                    if want_hidden and hs is not None:
                        if vis is not None and vstate["dim"] is None:
                            vstate["dim"] = int(vis.shape[0])
                        feat_buf[(arm, order_id)][0].append(sid)
                        feat_buf[(arm, order_id)][1].append(hs)
                        feat_buf[(arm, order_id)][2].append(vis)
                    n_done += 1
                    if n_done % 200 == 0:
                        el = time.time() - t0
                        print(f"[{worker}] {n_done} done  {el/max(1,n_done):.3f}s/it "
                              f"eta={(len(ids)*len(arms)*len(orders)-n_done)*el/max(1,n_done)/3600:.2f}h",
                              flush=True)
                except Exception as exc:
                    if "out of memory" in str(exc).lower():
                        try:
                            runner.torch.cuda.empty_cache()
                        except Exception:
                            pass
                    Q.append_jsonl(raw_path, {
                        **base_record(item, arm, order_id, args.model, revision,
                                      runner.proc_info, worker, args.device,
                                      prompt_type),
                        "status": "error", "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-1500:],
                        "resume_key": f"error|{arm}|{order_id}|{sid}"})
            # flush features for this arm/order
            flush((arm, order_id))

    print(f"[{worker}] finished {n_done} new records in "
          f"{(time.time()-t0)/3600:.2f}h", flush=True)
    try:
        peak = runner.torch.cuda.max_memory_allocated() / 1024 ** 2
        reserved = runner.torch.cuda.max_memory_reserved() / 1024 ** 2
        print(f"[{worker}] GPU peak allocated={peak:.0f}MiB reserved={reserved:.0f}MiB",
              flush=True)
        Q.write_json(Q.ROOT / "runtime" / f"gpu_peak_{worker}.json", {
            "worker": worker, "model": args.model,
            "peak_allocated_mib": round(peak, 1),
            "peak_reserved_mib": round(reserved, 1),
            "soft_limit_mib": args.gpu_soft_limit_mib,
            "n_records": n_done, "time": Q.now()})
    except Exception:
        pass


if __name__ == "__main__":
    main()
