#!/usr/bin/env python
"""Q6 -- Level 3: causal interventions on selected phenotypes.

TWO INTERVENTIONS, DELIBERATELY DIFFERENT IN KIND
-------------------------------------------------
1. CONTENT PATCH (bidirectional, activation-level).
   Item x is run twice on the same prompt: once with its correct image, once with
   a donor image.  The donor's decision-position state at row L is written into
   the recipient's forward pass at the same row, and the gold margin is re-read.
   Delta_M therefore measures how much of the image-condition difference is
   carried at that row and flows to the output.

   Donor states are read from the stored features rather than recomputed, so a
   patch costs one forward pass and no extra memory for a second model pass.

2. VISUAL KNOCKOUT (removal, token-level).
   The hidden states at the positions of the IMAGE tokens are zeroed at row L, so
   every layer after L runs without the visual stream.  "Direct" zeroes at every
   row; "relay" zeroes at one row only, which leaves the later layers free to
   re-derive something from the text stream.  The gap between the two is the part
   of the effect that survives re-derivation, and it is the reason the
   distinction is reported instead of a single number.

CONTROLS THAT MUST PASS BEFORE ANY NUMBER IS QUOTED
---------------------------------------------------
  identity   patch row L with the recipient's own state.  This does not merely
             approximate zero, it must BE zero: the patched tensor is the tensor
             the model would have produced.  Anything else means the hook is
             writing to the wrong row, and every effect below is then meaningless.
  random     patch with another item's state drawn from the same arm distribution.
Causal statements here are WITHIN one checkpoint.  No activation is ever moved
between checkpoints, and none of this licenses a cross-model causal claim.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_qr as Q   # noqa: E402

Q.cuda_alloc_env()


# --------------------------------------------------------------------------- #
def load_states(model: str, arm: str) -> dict:
    """Map sample_id -> (rows, hidden) for one arm, merged across item shards.

    Feature files are written per (arm, order, worker) because shards run in
    parallel; looking for `<arm>__O0.npz` finds nothing and the interventions
    then skip every donor state silently.
    """
    out: dict = {}
    d = Q.ROOT / "features" / model
    for p in sorted(d.glob(f"{arm}__O0__*.npz")):
        if p.name.endswith(".tmp.npz"):
            continue
        try:
            z = np.load(p, allow_pickle=False)
        except Exception:
            continue
        h = z["hidden"]
        for i, s in enumerate(z["sample_ids"]):
            out[str(s)] = h[i]
    return out


def pick_phenotypes(model: str, beh: dict, n_per: int) -> dict:
    """Items where the model failed in a way that has a natural causal question.

    Selection uses only the model's own behaviour on conditions it was not asked
    to be right about; it never uses the intervention outcome (that would be
    selecting on the result).
    """
    items = Q.load_items()
    ids = sorted(items)
    groups = {"semantic_conflict_failure": [], "severe_blur_failure": [],
              "plausible_mismatch_failure": [], "robust_correct": []}
    for sid in ids:
        c = beh.get((model, sid, "clean"))
        if c is None:
            continue
        clean_ok = c["acc"] > 0.5
        sc = beh.get((model, sid, "sc_clean"))
        bl = beh.get((model, sid, "blur_severe"))
        pm = beh.get((model, sid, "pm_clean_k1"))
        if clean_ok and sc is not None and sc["acc"] <= 0.5:
            groups["semantic_conflict_failure"].append(sid)
        elif clean_ok and bl is not None and bl["acc"] <= 0.5:
            groups["severe_blur_failure"].append(sid)
        elif clean_ok and pm is not None and pm["acc"] <= 0.5:
            groups["plausible_mismatch_failure"].append(sid)
        elif clean_ok:
            groups["robust_correct"].append(sid)
    rng = np.random.default_rng(20260917)
    out = {}
    for g, v in groups.items():
        v = sorted(v)
        rng.shuffle(v)
        out[g] = v[:n_per]
    return out


# --------------------------------------------------------------------------- #
class Patcher:
    """Runs one model with a hook that overwrites a single row's last position."""

    def __init__(self, model_label: str, device: str, attn: str):
        import torch
        from transformers import AutoConfig, AutoProcessor
        import transformers
        self.torch = torch
        self.device = device
        path = Q.MODELS[model_label]
        cls = getattr(transformers, AutoConfig.from_pretrained(path).architectures[0])
        self.model = cls.from_pretrained(path, torch_dtype="bfloat16",
                                         attn_implementation=attn,
                                         device_map=None).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(path)
        ip = self.processor.image_processor
        ip.min_pixels = Q.MATCHED_MIN_PIXELS
        ip.max_pixels = Q.MATCHED_MAX_PIXELS
        if hasattr(ip, "size") and isinstance(getattr(ip, "size"), dict):
            ip.size["longest_edge"] = Q.MATCHED_MAX_PIXELS
            ip.size["shortest_edge"] = Q.MATCHED_MIN_PIXELS
        self.tok = self.processor.tokenizer
        # The decoder stack sits under `language_model`; `model.model.layers` does
        # not exist and silently produced an AttributeError at patch time.
        base = self.model.model
        self.layers = (base.language_model.layers if hasattr(base, "language_model")
                       else base.layers)
        self.n_layers = len(self.layers)
        cfg = self.model.config
        self.image_token_id = (getattr(cfg, "image_token_id", None)
                               or getattr(getattr(cfg, "vision_config", None),
                                          "image_token_id", None)
                               or 151655)

    def build(self, prompt: str, image_path):
        from qwen_vl_utils import process_vision_info
        if image_path is None:
            msg = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True)
            return self.processor(text=[text], padding=True,
                                  return_tensors="pt").to(self.device), text
        msg = [{"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True)
        ii, vi = process_vision_info([msg])
        enc = self.processor(text=[text], images=ii, videos=vi, padding=True,
                             return_tensors="pt").to(self.device)
        return enc, text

    def run(self, enc, letters, patch_row=None, patch_vec=None, vis_pos=None,
            zero_vis=False, zero_from=None):
        """One forward with an optional intervention.

        zero_vis + zero_from: remove the visual stream from `zero_from` onward,
        which is the knockout family; zero_vis alone touches only `patch_row`.
        """
        torch = self.torch
        handles = []

        def make_hook(row):
            def hook(_m, _i, out):
                hs = out[0] if isinstance(out, (tuple, list)) else out
                hs = hs.clone()
                if vis_pos is not None and vis_pos.numel() and zero_vis:
                    if row >= zero_from:
                        hs[0, vis_pos, :] = 0.0
                if patch_row is not None and row == patch_row and patch_vec is not None:
                    hs[0, -1, :] = patch_vec
                if isinstance(out, tuple):
                    return (hs,) + out[1:]
                return hs
            return hook

        for i, layer in enumerate(self.layers):
            handles.append(layer.register_forward_hook(make_hook(i + 1)))
        try:
            with torch.inference_mode():
                logits = self.model(**enc, use_cache=False).logits[0, -1].float()
        finally:
            for h in handles:
                h.remove()
        lp = torch.log_softmax(logits, dim=-1)
        return {L: float(lp[self.tok.encode(L, add_special_tokens=False)[0]])
                for L in letters}

    def visual_positions(self, enc):
        """Positions of the image tokens in the decoder input.

        These are the token positions whose hidden states carry the projected
        visual stream; zeroing them is what "knockout" means here.
        """
        tid = self.image_token_id
        if tid is None:
            return None
        ids = enc["input_ids"][0]
        return self.torch.nonzero(ids == tid, as_tuple=False).flatten()


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(Q.MODELS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-per-phenotype", type=int, default=12)
    ap.add_argument("--layers", default="", help="comma rows; default = probe peak")
    ap.add_argument("--n-random-controls", type=int, default=2)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--gpu-soft-limit-mib", type=int, default=7000)
    args = ap.parse_args()

    out_dir = Q.ROOT / "results" / "causal"
    out_dir.mkdir(parents=True, exist_ok=True)
    man_dir = Q.ROOT / "manifests"
    man_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"CAUSAL__{args.model}.jsonl"

    from q2_behavior import load_behavior
    beh = load_behavior([args.model])
    if not beh:
        print("no behaviour yet -- nothing to select from")
        return
    pheno = pick_phenotypes(args.model, beh, args.n_per_phenotype)
    print({k: len(v) for k, v in pheno.items()})
    Q.write_json(man_dir / f"causal_subset_{args.model}.json", {
        "model": args.model, "phenotypes": pheno, "time": Q.now(),
        "selection_rule": ("clean-correct items whose outcome flips in the named "
                           "condition; selection never uses the intervention "
                           "outcome")})

    # layers: default to the rows the Level-1 mismatch probe found most readable
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        layers = [4, 8, 12, 16, 20, 24, 27]

    states = {}
    for arm in ("clean", "blur_severe", "pm_clean_k1", "pm_severe_k1",
                "sc_clean", "sc_severe"):
        states[arm] = load_states(args.model, arm)
        print(f"  states[{arm}] = {len(states[arm])}")

    import torch
    if args.gpu_soft_limit_mib > 0:
        idx = int(str(args.device).split(":")[-1])
        frac = min(0.95, args.gpu_soft_limit_mib * 1024 * 1024
                   / torch.cuda.get_device_properties(idx).total_memory)
        torch.cuda.set_per_process_memory_fraction(frac, idx)

    P = Patcher(args.model, args.device, args.attn)
    done = set()
    if out_path.is_file():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["resume_key"])

    items = Q.load_items()
    all_ids = sorted(items)
    rng = np.random.default_rng(20260917)
    n_written = 0
    t0 = time.time()

    # (condition label, donor arm) pairs; all within one checkpoint
    contrast = [("blur", "blur_severe"), ("mismatch", "pm_clean_k1"),
                ("semantic_conflict", "sc_clean")]

    for gname, sids in pheno.items():
        for sid in sids:
            item = items[sid]
            ro = Q.apply_order(item, "O0")
            letters = ro["display_letters"]
            d2s = ro["display_to_semantic"]
            gt = str(item["gt_index"])
            prompt = Q.build_m2_prompt(ro["mcq_text"])
            base_img = Q.image_path("clean", item)
            enc, text = P.build(prompt, base_img)
            vis_pos = P.visual_positions(enc)
            base = P.run(enc, letters)
            base_probs = Q.normalise_probs(base)
            base_sem = {str(d2s[L]): base_probs[L] for L in letters}
            base_M = Q.gt_margin({str(d2s[L]): base[L] for L in letters}, gt)

            own = states.get("clean", {}).get(sid)

            # `identity` and the two knockouts do not depend on which donor arm is
            # being compared, so they are evaluated once per (item, layer) rather
            # than once per contrast.  Repeating them would triple the GPU cost to
            # reproduce the same control number three times.
            todo = []
            for L in layers:
                todo.append((L, "identity", None))
                todo.append((L, "visual_knockout_direct", None))
                todo.append((L, "visual_knockout_relay", None))
                for cname, donor_arm in contrast:
                    todo.append((L, f"content_patch::{cname}", donor_arm))
                    if args.n_random_controls:
                        todo.append((L, f"random_patch::{cname}", donor_arm))

            for L, kind, donor_arm in todo:
                cname = kind.split("::")[1] if "::" in kind else "shared"
                base_kind = kind.split("::")[0]
                key = Q.stable_seed("causal", args.model, gname, sid, cname, L,
                                    base_kind, Q.MATCHED_PROCESSOR_ID)
                if key in done:
                    continue
                try:
                    if base_kind == "identity":
                        if own is None or L >= own.shape[0]:
                            continue
                        pv = torch.tensor(own[L], dtype=torch.bfloat16,
                                          device=args.device)
                        lg = P.run(enc, letters, patch_row=L, patch_vec=pv)
                    elif base_kind == "visual_knockout_direct":
                        if vis_pos is None or not vis_pos.numel():
                            continue
                        lg = P.run(enc, letters, vis_pos=vis_pos, zero_vis=True,
                                   zero_from=0)
                    elif base_kind == "visual_knockout_relay":
                        if vis_pos is None or not vis_pos.numel():
                            continue
                        lg = P.run(enc, letters, vis_pos=vis_pos, zero_vis=True,
                                   zero_from=L)
                    elif base_kind == "content_patch":
                        dst = states.get(donor_arm, {}).get(sid)
                        if dst is None or L >= dst.shape[0]:
                            continue
                        pv = torch.tensor(dst[L], dtype=torch.bfloat16,
                                          device=args.device)
                        lg = P.run(enc, letters, patch_row=L, patch_vec=pv)
                    else:                                   # random_patch
                        dst = states.get("clean", {})
                        if own is None or L >= own.shape[0]:
                            continue
                        pv = torch.tensor(_random_vec(dst, all_ids, sid, L, rng),
                                          dtype=torch.bfloat16, device=args.device)
                        lg = P.run(enc, letters, patch_row=L, patch_vec=pv)
                except Exception as exc:
                    Q.append_jsonl(out_path, {
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-1200:],
                        "model": args.model, "sample_id": sid,
                        "phenotype": gname, "contrast": cname, "layer": L,
                        "intervention": base_kind, "resume_key": key,
                        "timestamp": Q.now()})
                    continue
                pr = Q.normalise_probs(lg)
                sem = {str(d2s[L2]): pr[L2] for L2 in letters}
                sem_log = {str(d2s[L2]): lg[L2] for L2 in letters}
                M = Q.gt_margin(sem_log, gt)
                Q.append_jsonl(out_path, {
                    "status": "ok", "model": args.model,
                    "sample_id": sid, "modality": item["modality"],
                    "phenotype": gname, "contrast": cname, "layer": L,
                    "intervention": base_kind,
                    "donor_arm": donor_arm if base_kind == "content_patch" else None,
                    "M_base": base_M, "M_intervened": M, "delta_M": M - base_M,
                    "p_gt_base": base_sem.get(gt),
                    "p_gt_intervened": sem.get(gt),
                    "correct_base": bool((base_sem.get(gt) or 0)
                                         == max(base_sem.values())),
                    "correct_intervened": bool(sem.get(gt) == max(sem.values())),
                    "js_vs_base": Q.js_divergence(base_sem, sem),
                    "n_visual_positions": int(vis_pos.numel())
                    if vis_pos is not None else 0,
                    "processor_id": Q.MATCHED_PROCESSOR_ID,
                    "prompt_hash": Q.sha256_text(text),
                    "resume_key": key, "timestamp": Q.now()})
                n_written += 1
                if n_written % 50 == 0:
                    el = time.time() - t0
                    print(f"  {n_written} interventions, "
                          f"{el/max(1,n_written):.2f}s each", flush=True)
    print(f"wrote {n_written} causal records to {out_path}")


def _random_vec(states: dict, ids: list, sid: str, L: int, rng) -> np.ndarray:
    for _ in range(50):
        other = ids[int(rng.integers(0, len(ids)))]
        if other == sid:
            continue
        s = states.get(other)
        if s is not None and L < s.shape[0]:
            return s[L]
    return np.zeros(1)


if __name__ == "__main__":
    main()
