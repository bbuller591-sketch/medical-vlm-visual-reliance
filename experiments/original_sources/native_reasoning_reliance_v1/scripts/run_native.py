#!/usr/bin/env python
from __future__ import annotations
import argparse, hashlib, json, math, os, re, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MED = ROOT.parent
QR_SCRIPTS = MED / "quality_reliance_v1" / "scripts"
sys.path.insert(0, str(QR_SCRIPTS))
import common_qr as Q

Q.cuda_alloc_env()

OFFICIAL_TEMPLATE = """{problem}
Your task:
1. Think through the question step by step, enclose your reasoning process in <think>...</think> tags.
2. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags.
3. No extra information or text outside of these tags.
"""

STRUCTURED_TEMPLATE = """{problem}
Your task:
1. Reason through the medical image and question step by step.
2. Put only the reasoning process inside <think>...</think>.
3. Do not output the final answer yet. The opening <think> tag is already supplied; finish the reasoning by outputting </think>.
"""

SIMPLE_TEMPLATE = """{problem}
Your task: provide the correct single-letter choice (A, B, C, D,...).
"""

COMMON_TEMPLATE = """{problem}
Reason through the medical image and question step by step. After the reasoning, give exactly one final option letter in the form:
FINAL: X
where X is one of the listed option letters. Do not write anything after the final letter.
"""
COMMON_RE = re.compile(r"(?s)(.*?\bFINAL:\s*)([A-J])\s*\Z")

STRICT_RE = re.compile(r"(?s)\s*(<think>.*?</think>)\s*<answer>\s*([A-J])\s*</answer>\s*\Z")

def stable_int(*parts: str) -> int:
    s = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(s).digest()[:8], "big")

def make_noise_pil(path: Path, sample_id: str, max_pixels: int):
    from PIL import Image
    import numpy as np
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = min(1.0, math.sqrt(max_pixels / max(1, w*h)))
    nw, nh = max(8, int(round(w*scale))), max(8, int(round(h*scale)))
    im = im.resize((nw, nh), Image.Resampling.BILINEAR)
    a = np.asarray(im, dtype=np.float32)
    mean = a.reshape(-1,3).mean(0)
    std = a.reshape(-1,3).std(0)
    std = np.maximum(std, 12.0)
    rng = np.random.default_rng(stable_int("pure_noise", sample_id))
    n = rng.normal(mean, std, size=a.shape)
    n = np.clip(n, 0, 255).astype(np.uint8)
    return Image.fromarray(n, "RGB"), {"noise_width": nw, "noise_height": nh,
        "source_mean_rgb": mean.round(3).tolist(), "source_std_rgb": std.round(3).tolist()}

class NativeRunner:
    def __init__(self, model_label: str, device: str, attn: str="sdpa",
                 min_pixels: int=Q.MATCHED_MIN_PIXELS, max_pixels: int=Q.MATCHED_MAX_PIXELS):
        import torch, transformers
        from transformers import AutoConfig, AutoProcessor
        self.torch = torch
        self.label = model_label
        self.device = device
        path = Q.MODELS[model_label]
        cls = getattr(transformers, AutoConfig.from_pretrained(path).architectures[0])
        self.model = cls.from_pretrained(path, torch_dtype="bfloat16",
            attn_implementation=attn, device_map=None).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(path)
        ip = self.processor.image_processor
        ip.min_pixels = min_pixels
        ip.max_pixels = max_pixels
        if hasattr(ip, "size") and isinstance(ip.size, dict):
            ip.size["shortest_edge"] = min_pixels
            ip.size["longest_edge"] = max_pixels
        self.tok = self.processor.tokenizer
        self.min_pixels, self.max_pixels = min_pixels, max_pixels

    def _message(self, prompt: str, image):
        if image is None:
            return [{"role":"user","content":[{"type":"text","text":prompt}]}]
        image_value = f"file://{Path(image)}" if isinstance(image, (str, Path)) else image
        return [{"role":"user","content":[{"type":"image","image":image_value},
                                           {"type":"text","text":prompt}]}]

    def _encode_message(self, prompt: str, image):
        msg = self._message(prompt, image)
        text = self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        if image is None:
            enc = self.processor(text=[text], padding=True, return_tensors="pt").to(self.device)
            return enc, text, None, None
        if isinstance(image, (str, Path)):
            from qwen_vl_utils import process_vision_info
            ii, vi = process_vision_info([msg])
            enc = self.processor(text=[text], images=ii, videos=vi, padding=True,
                                 return_tensors="pt").to(self.device)
            return enc, text, ii, vi
        # PIL image: processor can consume it directly; chat template only needs image marker.
        enc = self.processor(text=[text], images=[image], padding=True,
                             return_tensors="pt").to(self.device)
        return enc, text, [image], None

    def generate_and_score(self, prompt: str, image, letters: list[str], max_new_tokens: int=512,
                           mode: str="structured"):
        enc, chat_text, ii, vi = self._encode_message(prompt, image)
        generated_letter = None
        if mode == "simple":
            # Paper-faithful Qwen baseline policy: score the next bare option letter
            # directly from the prompt, then separately generate the bare answer for
            # compliance/accuracy diagnostics. This is NOT a reasoning-conditioned arm.
            with self.torch.inference_mode():
                pre = self.model(**enc, use_cache=False)
            pre_logits = pre.logits[0,-1].float()
            lpv = self.torch.log_softmax(pre_logits, dim=-1)
            cand_lp = {}
            token_modes = {}
            for L in letters:
                ids = self.tok.encode(L, add_special_tokens=False)
                if len(ids) != 1:
                    return {"status":"candidate_tokenization_fail","candidate":L,"candidate_token_ids":ids}
                cand_lp[L] = float(lpv[ids[0]])
                token_modes[L] = ids[0]
            z=max(cand_lp.values()); den=sum(math.exp(v-z) for v in cand_lp.values())
            probs={k:math.exp(v-z)/den for k,v in cand_lp.items()}
            input_len=int(enc.input_ids.shape[1])
            with self.torch.inference_mode():
                g=self.model.generate(**enc,use_cache=True,max_new_tokens=max_new_tokens,do_sample=False)
            new_ids=g[0,input_len:]
            generated=self.processor.decode(new_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            gen=generated.strip()
            ok=bool(re.fullmatch(r"[A-J]",gen))
            base={"generated_text":generated,"format_ok":ok,"n_generated_tokens":int(new_ids.numel()),
                  "reasoning_mode":"paper_faithful_simple"}
            if not ok:
                return {**base,"status":"format_fail"}
            top=max(probs,key=probs.get)
            return {**base,"status":"ok","reasoning_prefix":None,"generated_answer_letter":gen,
                    "candidate_logprobs":cand_lp,"candidate_probs":probs,
                    "pred_letter_from_probs":top,"prob_top1_matches_generated":bool(top==gen),
                    "candidate_token_ids":token_modes,
                    "n_visual_tokens": int((enc.get("image_grid_thw").prod(-1) //
                        ((getattr(self.processor.image_processor,"merge_size",1) or 1)**2)).sum())
                        if enc.get("image_grid_thw") is not None else 0}
        if mode in ("native", "common"):
            input_len = int(enc.input_ids.shape[1])
            with self.torch.inference_mode():
                g = self.model.generate(**enc, use_cache=True, max_new_tokens=max_new_tokens,
                                        do_sample=False)
            new_ids = g[0, input_len:]
            generated = self.processor.decode(new_ids, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)
            m = STRICT_RE.fullmatch(generated) if mode=="native" else COMMON_RE.fullmatch(generated)
            base = {"generated_text": generated, "format_ok": bool(m),
                    "n_generated_tokens": int(new_ids.numel()),
                    "reasoning_mode":("native_strict" if mode=="native" else "common_reasoning_final")}
            if not m:
                return {**base, "status":"format_fail"}
            if mode == "native":
                think_block = m.group(1)
                # Canonical answer scaffold prevents BPE fusion such as ">A" from
                # contaminating the candidate comparison. No answer letter is present.
                prefix = think_block + "\n\n<answer>\n"
                generated_letter = m.group(2)
            else:
                prefix = m.group(1)
                generated_letter = m.group(2)
        else:
            # Fair Base-vs-RL instrument: force only the structural opening tag,
            # let the checkpoint generate its own reasoning, require it to close
            # </think>, then score the answer letters at a common <answer> prefix.
            opening = "<think>\n"
            if image is None:
                gen_enc = self.processor(text=[chat_text + opening], padding=True,
                                         return_tensors="pt").to(self.device)
            else:
                gen_enc = self.processor(text=[chat_text + opening], images=ii, videos=vi,
                                         padding=True, return_tensors="pt").to(self.device)
            input_len = int(gen_enc.input_ids.shape[1])
            with self.torch.inference_mode():
                g = self.model.generate(**gen_enc, use_cache=True,
                                        max_new_tokens=max_new_tokens, do_sample=False)
            new_ids = g[0, input_len:]
            continuation = self.processor.decode(new_ids, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)
            end = continuation.find("</think>")
            base = {"generated_text": opening + continuation, "format_ok": end >= 0,
                    "n_generated_tokens": int(new_ids.numel()),
                    "reasoning_mode":"structured_prefill"}
            if end < 0:
                return {**base, "status":"format_fail"}
            reasoning = continuation[:end]
            prefix = opening + reasoning + "</think>\n<answer>\n"

        full_text = chat_text + prefix
        if image is None:
            sc = self.processor(text=[full_text], padding=True, return_tensors="pt").to(self.device)
        else:
            sc = self.processor(text=[full_text], images=ii, videos=vi, padding=True,
                                return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            out = self.model(**sc, use_cache=False)
        logits = out.logits[0,-1].float()
        logp_vocab = self.torch.log_softmax(logits, dim=-1)
        cand_lp = {}
        token_modes = {}
        for L in letters:
            ids = self.tok.encode(L, add_special_tokens=False)
            if len(ids) != 1:
                # For this benchmark A-J are expected to be single tokens. Do not
                # silently approximate a multi-token answer with the first token.
                return {**base, "status":"candidate_tokenization_fail",
                        "candidate":L, "candidate_token_ids":ids}
            cand_lp[L] = float(logp_vocab[ids[0]])
            token_modes[L] = ids[0]
        z = max(cand_lp.values())
        denom = sum(math.exp(v-z) for v in cand_lp.values())
        probs = {k: math.exp(v-z)/denom for k,v in cand_lp.items()}
        top = max(probs, key=probs.get)
        return {
            **base, "status":"ok", "reasoning_prefix":prefix,
            "generated_answer_letter":generated_letter,
            "candidate_logprobs":cand_lp, "candidate_probs":probs,
            "pred_letter_from_probs":top,
            "prob_top1_matches_generated": (None if generated_letter is None else bool(top == generated_letter)),
            "candidate_token_ids":token_modes,
            "n_visual_tokens": int((sc.get("image_grid_thw").prod(-1) //
                ((getattr(self.processor.image_processor,"merge_size",1) or 1)**2)).sum())
                if sc.get("image_grid_thw") is not None else 0,
        }

def resolve_image(arm: str, item: dict, max_pixels: int):
    if arm == "pure_noise":
        src = Q.source_image_path(item)
        return make_noise_pil(src, item["sample_id"], max_pixels)
    if arm == "blur_mid_extra":
        p = MED / "official900_visual_reliance_v1" / "generated_images" / "blur_mid_extra" / f"{item['sample_id']}.png"
        return (p, {}) if p.is_file() else (None, {})
    p = Q.image_path(arm, item)
    return (p, {}) if p is not None else (None, {})

def arm_spec(arm: str):
    if arm == "pure_noise":
        return ("uninterpretable", "pure_noise")
    if arm == "blur_mid_extra":
        return ("mid_extra", "matched")
    return Q.ARM_SPEC.get(arm, (None, arm))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True,choices=sorted(Q.MODELS))
    ap.add_argument("--device",default="cuda:0")
    ap.add_argument("--arms",default="clean,no_image,pm_clean_k1,sc_clean,pure_noise")
    ap.add_argument("--shard",type=int,default=0)
    ap.add_argument("--nshards",type=int,default=1)
    ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--max-new-tokens",type=int,default=512)
    ap.add_argument("--mode",choices=["structured","native","common","simple"],default="common")
    ap.add_argument("--worker",default="")
    args=ap.parse_args()

    items=Q.load_items()
    ids=sorted(items)
    if args.limit: ids=ids[:args.limit]
    ids=ids[args.shard::args.nshards]
    arms=[x for x in args.arms.split(",") if x]
    worker=args.worker or f"{args.model}_s{args.shard}"
    outdir=ROOT/"results"/"raw"; outdir.mkdir(parents=True,exist_ok=True)
    outp=outdir/f"NATIVE__{worker}.jsonl"
    done=set()
    if outp.exists():
        for line in outp.open():
            try:
                r=json.loads(line); done.add((r["sample_id"],r["arm"]))
            except Exception: pass

    print(f"[{worker}] model={args.model} items={len(ids)} arms={arms} existing={len(done)}",flush=True)
    runner=NativeRunner(args.model,args.device)
    ok=ff=err=0; t0=time.time()
    with outp.open("a",encoding="utf-8") as fw:
        for arm in arms:
            for sid in ids:
                if (sid,arm) in done: continue
                item=items[sid]
                qlevel, rel=arm_spec(arm)
                rec={"sample_id":sid,"file_index":item["file_index"],"modality":item["modality"],
                    "model":args.model,"arm":arm,"quality_level":qlevel,"relevance_type":rel,
                    "gt_index":item["gt_index"],"gt_letter":item["gt_letter"],"gt_text":item["gt_text"],
                    "n_options":item["n_options"],"timestamp":Q.now(),
                    "processor_id":Q.MATCHED_PROCESSOR_ID,
                    "prompt_type":({"native":"native_reasoning_reward_aligned_v1",
                                    "structured":"structured_reasoning_prefill_v1",
                                    "common":"common_reasoning_final_v1",
                                    "simple":"paper_faithful_simple_v1"}[args.mode])}
                try:
                    image, extra=resolve_image(arm,item,Q.MATCHED_MAX_PIXELS)
                    if arm not in ("no_image","pure_noise") and image is None:
                        rec.update(status="missing_donor")
                    else:
                        template = (OFFICIAL_TEMPLATE if args.mode=="native" else
                                    (COMMON_TEMPLATE if args.mode=="common" else
                                     (SIMPLE_TEMPLATE if args.mode=="simple" else STRUCTURED_TEMPLATE)))
                        prompt=template.format(problem=item["problem"])
                        rr=runner.generate_and_score(prompt,image,item["letters"],args.max_new_tokens,args.mode)
                        rec.update(extra); rec.update(rr)
                        if rr["status"]=="ok":
                            probs=rr["candidate_probs"]; lp=rr["candidate_logprobs"]
                            gt=item["gt_letter"]
                            wrong=[L for L in item["letters"] if L!=gt]
                            rec["p_gt"]=probs[gt]
                            rec["gt_log_margin"]=lp[gt]-max(lp[L] for L in wrong)
                            rec["gt_prob_margin"]=probs[gt]-max(probs[L] for L in wrong)
                            rec["entropy"]=-sum(p*math.log(max(p,1e-15)) for p in probs.values())
                            rec["generated_correct"]=(None if rr.get("generated_answer_letter") is None
                                                      else rr["generated_answer_letter"]==gt)
                            rec["prob_top1_correct"]=rr["pred_letter_from_probs"]==gt
                            ok+=1
                        elif rr["status"]=="format_fail": ff+=1
                except Exception as e:
                    rec.update(status="error",error=f"{type(e).__name__}: {e}",
                               traceback=traceback.format_exc()[-2500:]); err+=1
                    try: runner.torch.cuda.empty_cache()
                    except Exception: pass
                fw.write(json.dumps(rec,ensure_ascii=False)+"\n"); fw.flush()
                n=ok+ff+err
                if n and n%25==0:
                    el=time.time()-t0
                    print(f"[{worker}] processed={n} ok={ok} format_fail={ff} err={err} rate={n/el:.3f}/s",flush=True)
    print(json.dumps({"worker":worker,"ok":ok,"format_fail":ff,"errors":err,
                      "elapsed_min":(time.time()-t0)/60}),flush=True)

if __name__=="__main__":
    main()
