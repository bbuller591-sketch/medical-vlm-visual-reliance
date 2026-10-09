#!/usr/bin/env python
"""Paper-faithful clean baseline replication of the official MedVLM-R1 900 MCQ.

WHAT THIS IS
A line-by-line port of the official evaluation entry point,

    sources/MedVLM-R1_repo/src/eval/test_qwen2vl_med.py
    driven by sources/MedVLM-R1_repo/test_script.sh

on the official test file

    sources/MedVLM-R1_repo/MRI_CT_XRAY_300each_dataset.json

Nothing here is a design choice: prompt construction, chat-template call, image
handling, generation call, answer extraction and the accuracy denominator are all
reproduced from the official source. Where this implementation must differ, the
difference is recorded in ``DEVIATIONS`` below and surfaced in the output record.

WHAT THIS IS NOT
No blur, noise, no-image, mismatch, paraphrase, probing, patching, calibration or
probability instrument. Clean images only. The single question is whether the
official 900 clean baseline reproduces.

DEVIATIONS FROM THE OFFICIAL SCRIPT (all recorded per run)
  1. ``attn_implementation``: the official script requests ``flash_attention_2``.
     The ``flash_attn`` package is not installed in this environment, and installing
     it would risk disturbing the shared ``medvlm-reliance`` env used by other
     projects, so ``sdpa`` is used. This affects speed and float summation order,
     not the prompt, the decoding rule or the parser. It is recorded on every run
     and is the first thing to revisit if the result lands in the RED band.
  2. ``device_map="auto"`` is replaced by an explicit device when one is given, so
     two model runs can be pinned to two GPUs.
  3. The official script hard-codes ``max_new_tokens=512`` inside ``generate()``
     while also passing a ``generation_config``; both are reproduced verbatim, and
     the value is recorded.

GENERATION-CONFIG CONFLICT (deliberately not resolved silently)
  * ``models/MedVLM-R1/generation_config.json`` (shipped in the checkpoint) says
    ``do_sample=true, temperature=0.01, top_k=1, top_p=0.001``.
  * The HF model card Quick Start says ``do_sample=False, max_new_tokens=1024``.
  * ``test_script.sh`` says ``MAX_NEW_TOKENS=512, DO_SAMPLE=false, TEMPERATURE=1.0``.
  The official script passes an explicit ``GenerationConfig`` and an explicit
  ``do_sample=False`` kwarg, which overrides the checkpoint file. That is what this
  runner does. All three sources are saved under ``configs/`` and the choice is
  reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

# Must be set BEFORE torch creates its CUDA context.
#
# WHY: the official config keeps Qwen2-VL's own ``max_pixels`` (12845056), so a
# handful of full-resolution radiographs (3.9% of the 900) expand to ~16k visual
# tokens. The vision-encoder activations for one such image peak around 22 GiB on a
# 24 GB card, and without this setting the run died with
#   "Tried to allocate 7.84 GiB ... 4.30 GiB is reserved by PyTorch but unallocated"
# i.e. the memory was there but fragmented. expandable_segments makes the reserved
# pool reusable, which lets the OFFICIAL preprocessing be kept rather than capping
# ``max_pixels`` and changing the model's input.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------------- #
# Verbatim from src/eval/test_qwen2vl_med.py (the `complex` branch).
# Whitespace, trailing spaces and the doubled space at the end of line 1 are
# preserved exactly as in the official file.
# --------------------------------------------------------------------------- #
QUESTION_TEMPLATE_COMPLEX = """
    {Question} 
    Your task: 
    1. Think through the question step by step, enclose your reasoning process in <think>...</think> tags. 
    2. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags.
    3. No extra information or text outside of these tags.
    """

QUESTION_TEMPLATE_SIMPLE = """
    {Question} 
    Your task: provide the correct single-letter choice (A, B, C, D,...).
    """

# Verbatim from src/eval/test_qwen2vl_med.py
ANSWER_RE = re.compile(r"<answer>\s*([A-Z])\s*</answer>")

MODALITY_SLICE = {"MRI": (0, 300), "CT": (300, 600), "Xray": (600, 900)}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def rec_id(it: dict) -> str:
    return f"{it['modality']}_{it['index_in_modality']:03d}"


def extract_letter_answer(output_str: str) -> str | None:
    """Official extractor, verbatim semantics."""
    m = ANSWER_RE.search(output_str or "")
    return m.group(1) if m else None


def think_content(text: str) -> str | None:
    m = re.search(r"<think>(.*?)</think>", text or "", re.S | re.I)
    return m.group(1).strip() if m else None


def answer_tag_content(text: str) -> str | None:
    m = re.search(r"<answer>(.*?)</answer>", text or "", re.S | re.I)
    return m.group(1) if m else None


def load_done_keys(path: Path) -> dict[str, dict]:
    """Resume support: key -> existing record."""
    done: dict[str, dict] = {}
    if path.is_file():
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("resume_key"):
                    done[r["resume_key"]] = r
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True,
                    help="local snapshot dir or HF id")
    ap.add_argument("--model-label", required=True,
                    help="short label used in output records, e.g. medvlmr1")
    ap.add_argument("--modalities", nargs="+", default=["MRI", "CT", "Xray"],
                    choices=["MRI", "CT", "Xray"])
    ap.add_argument("--prompt-type", default="complex", choices=["simple", "complex"])
    ap.add_argument("--test-json", default=str(
        ROOT / "sources" / "MedVLM-R1_repo" / "MRI_CT_XRAY_300each_dataset.json"))
    ap.add_argument("--base-path", required=True,
                    help="BASE_PATH in the official script: the directory that "
                         "contains the `images/` folder")
    ap.add_argument("--out", required=True, help="sample-level output JSONL")
    ap.add_argument("--summary", default=None, help="aggregate summary JSON")
    ap.add_argument("--max-new-tokens", type=int, default=512,
                    help="test_script.sh uses 512; the HF model card shows 1024. "
                         "Primary = 512 (official evaluation entry point).")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--attn", default="sdpa",
                    help="official script requests flash_attention_2; sdpa is used "
                         "here because flash_attn is not installed (recorded)")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: only the first N items per modality")
    ap.add_argument("--sample-per-modality", type=int, default=None,
                    help="smoke test: take N items per modality drawn at random "
                         "with a fixed seed, so the sample is stratified but not "
                         "just the first rows (which share a question template)")
    ap.add_argument("--sample-seed", type=int, default=20260913)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--dump-prompts", type=int, default=0,
                    help="write the fully rendered prompt for the first N items "
                         "to diagnostics/rendered_prompts/ (required for the "
                         "official-script cross-check)")
    args = ap.parse_args()

    import torch
    from transformers import (AutoProcessor, GenerationConfig,
                              Qwen2VLForConditionalGeneration)
    from qwen_vl_utils import process_vision_info

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = json.loads(Path(args.test_json).read_text())

    # ---- build the work list, exactly as the official runner slices it ------- #
    items = []
    for mod in args.modalities:
        lo, hi = MODALITY_SLICE[mod]
        block = list(enumerate(dataset[lo:hi]))
        if args.sample_per_modality is not None:
            import random
            # Seed from the modality's POSITION, not from `hash(mod)`: Python's
            # string hash is process-salted, so the same smoke sample would differ
            # between runs and could not be re-checked later.
            rng = random.Random(args.sample_seed + args.modalities.index(mod))
            block = sorted(rng.sample(block, min(args.sample_per_modality,
                                                 len(block))))
        if args.limit:
            block = block[:args.limit]
        for j, r in block:
            items.append({"modality": mod, "index_in_modality": j,
                          "file_index": lo + j, "row": r})

    template = (QUESTION_TEMPLATE_COMPLEX if args.prompt_type == "complex"
                else QUESTION_TEMPLATE_SIMPLE)

    # ---- model ---------------------------------------------------------------- #
    t0 = time.time()
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn,
        device_map=None,
    ).to(args.device).eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    load_s = time.time() - t0

    # The official script passes BOTH an explicit GenerationConfig and explicit
    # kwargs to generate(). Reproduced as-is.
    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=1.0,
        num_return_sequences=1,
        pad_token_id=151643,
    )
    gen_cfg_hash = sha256_text(json.dumps(gen_cfg.to_dict(), sort_keys=True,
                                         default=str))
    prompt_hash = sha256_text(template)

    ip = processor.image_processor
    run_provenance = {
        "model_path": args.model_path,
        "model_label": args.model_label,
        "model_revision": getattr(model.config, "_name_or_path", None),
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "dtype": "bfloat16",
        "device": args.device,
        "attn_implementation": args.attn,
        "attn_requested_by_official_script": "flash_attention_2",
        "prompt_type": args.prompt_type,
        "prompt_hash": prompt_hash,
        "prompt_template": template,
        "generation_config": gen_cfg.to_dict(),
        "generation_config_hash": gen_cfg_hash,
        "official_script_generate_kwargs": {"use_cache": True,
                                            "max_new_tokens": 512,
                                            "do_sample": False},
        "processor_class": type(processor).__name__,
        "image_processor_class": type(ip).__name__,
        "min_pixels": getattr(ip, "min_pixels", None),
        "max_pixels": getattr(ip, "max_pixels", None),
        "patch_size": getattr(ip, "patch_size", None),
        "merge_size": getattr(ip, "merge_size", None),
        "test_json": str(args.test_json),
        "test_json_sha256": sha256_file(Path(args.test_json)),
        "base_path": args.base_path,
    }

    done = load_done_keys(out_path)
    print(f"[official900] model={args.model_label} items={len(items)} "
          f"already_done={len(done)} load={load_s:.1f}s")
    print(f"[official900] attn={args.attn} max_new_tokens={args.max_new_tokens} "
          f"min/max_pixels={run_provenance['min_pixels']}/"
          f"{run_provenance['max_pixels']}")

    written = 0
    skipped = 0
    n_correct = 0
    n_parsed = 0
    t_start = time.time()

    # Rendered prompts are exported verbatim so they can be diffed against the
    # official script's own output for the same items (cross-check step).
    prompt_dir = ROOT / "diagnostics" / "rendered_prompts"
    if args.dump_prompts:
        prompt_dir.mkdir(parents=True, exist_ok=True)
    n_prompts_dumped = 0

    with open(out_path, "a", encoding="utf-8") as fh:
        for it in items:
            row = it["row"]
            image_rel = row["image"][0]
            gt_letter = row["solution"]
            img_path = Path(args.base_path) / image_rel

            img_hash = sha256_file(img_path) if img_path.is_file() else None
            resume_key = sha256_text("|".join([
                str(row.get("question")), "|".join(row.get("options") or []),
                image_rel, img_hash or "NOIMAGE",
                args.model_label, str(run_provenance.get("model_revision")),
                prompt_hash, gen_cfg_hash,
            ]))

            if resume_key in done:
                rec = done[resume_key]
                n_correct += int(bool(rec.get("correct")))
                n_parsed += int(bool(rec.get("parse_success")))
                skipped += 1
                continue

            # ---- messages: verbatim structure of the official runner ---------- #
            # Note: a `system` message is NOT added. `convert_example_SFT` in the
            # official file defines a SYSTEM_PROMPT but is never called; the loop
            # that builds `messages` emits a single user turn.
            message = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{args.base_path}/{image_rel}"},
                    {"type": "text",
                     "text": template.format(Question=row["problem"])},
                ],
            }]

            text = processor.apply_chat_template(message, tokenize=False,
                                                 add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info([message])
            inputs = processor(text=[text], images=image_inputs,
                               videos=video_inputs, padding=True,
                               return_tensors="pt").to(args.device)

            if n_prompts_dumped < args.dump_prompts:
                (prompt_dir /
                 f"{args.model_label}_{rec_id(it)}.txt").write_text(
                    text, encoding="utf-8")
                n_prompts_dumped += 1
            visual_tokens = int((inputs.get("image_grid_thw") is not None
                                 and int((inputs["image_grid_thw"].prod(-1)
                                          // (processor.image_processor.merge_size ** 2)
                                          ).sum()) or 0))

            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs, use_cache=True,
                    max_new_tokens=args.max_new_tokens, do_sample=False,
                    generation_config=gen_cfg,
                )
            trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, generated_ids)]
            gen_text = processor.batch_decode(
                trimmed, skip_special_tokens=True,
                clean_up_tokenization_spaces=False)[0]

            # ---- answer extraction: the official branch, reproduced exactly --- #
            #   if Prompt_type == 'complex': model_answer = extract_letter_answer(out)
            #   elif Prompt_type == 'simple': model_answer = original_output
            #   if model_answer is not None and model_answer == ground_truth: correct += 1
            # The `simple` branch is a RAW STRING comparison against the ground-truth
            # letter - it is not parsed at all. An earlier version of this runner
            # applied the <answer> regex in both modes, which scored every simple-mode
            # response as unparsed and reported 0% for a model that was in fact
            # emitting the bare letter.
            raw_output = gen_text
            if args.prompt_type == "complex":
                parsed = extract_letter_answer(raw_output)
            else:
                parsed = raw_output
            correct = (parsed is not None and parsed == gt_letter)

            # Diagnostics only - never the official metric for simple mode.
            bare = (str(parsed).strip() if parsed is not None else None)
            bare_is_letter = bool(bare and re.fullmatch(r"[A-J]", bare))
            correct_after_strip = (bare is not None and bare == gt_letter)

            rec = {
                "sample_id": f"{it['modality']}_{it['index_in_modality']:03d}",
                "file_index": it["file_index"],
                "model": args.model_label,
                "modality": it["modality"],
                "image": image_rel,
                "image_sha256": img_hash,
                "question": row["question"],
                "problem": row["problem"],
                "options": row["options"],
                "gt_semantic_answer": row["answer"],
                "gt_letter": gt_letter,
                "generated_text": gen_text,
                "answer_tag_content": answer_tag_content(gen_text),
                "think_content": think_content(gen_text),
                "parsed_letter": parsed,
                "parse_success": (parsed is not None
                                  if args.prompt_type == "complex"
                                  else bare_is_letter),
                "bare_output_is_single_letter": bare_is_letter,
                "correct_after_strip_diagnostic": correct_after_strip,
                "format_compliant": bool(
                    re.search(r"<think>.*?</think>", gen_text or "", re.S | re.I)
                    and parsed is not None),
                "correct": correct,
                "generated_tokens": int(trimmed[0].shape[0]),
                "prompt_tokens": int(inputs.input_ids.shape[1]),
                "visual_tokens": visual_tokens,
                "image_hw": [int(inputs["image_grid_thw"][0][1]) * 14,
                             int(inputs["image_grid_thw"][0][2]) * 14]
                if inputs.get("image_grid_thw") is not None else None,
                "resume_key": resume_key,
                "prompt_hash": prompt_hash,
                "generation_config_hash": gen_cfg_hash,
                "provenance": {k: v for k, v in run_provenance.items()
                               if k != "prompt_template"},
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            written += 1
            n_correct += int(correct)
            n_parsed += int(parsed is not None)

            # Release the batch and defragment after any unusually large image.
            # Some official-config items reach ~16k visual tokens and dominate the
            # peak; holding them while the next item allocates its own tensors is
            # how the run died before.
            n_visual = visual_tokens
            del inputs, generated_ids, trimmed
            if n_visual > 4000:
                torch.cuda.empty_cache()
                peak = torch.cuda.max_memory_allocated() / 2**30
                print(f"    [mem] item {rec['sample_id']} visual_tokens={n_visual} "
                      f"peak={peak:.2f}GiB", flush=True)

            if written % args.log_every == 0:
                el = time.time() - t_start
                rate = written / el if el else 0
                print(f"  [{args.model_label}] written={written} "
                      f"skipped={skipped} acc={n_correct/max(1,(written+skipped)):.4f} "
                      f"elapsed={el/60:.1f}min rate={rate:.2f}/s", flush=True)

    total = written + skipped
    # Official denominator: len(data), i.e. unparsed answers count as WRONG.
    summary = {
        "model": args.model_label,
        "model_path": args.model_path,
        "modalities": args.modalities,
        "n_total": total,
        "n_written_this_run": written,
        "n_resumed": skipped,
        "strict_accuracy": n_correct / total if total else None,
        "strict_accuracy_pct": (n_correct / total * 100) if total else None,
        "n_correct": n_correct,
        "parse_success_rate": n_parsed / total if total else None,
        "n_parsed": n_parsed,
        "official_accuracy_definition": "correct / len(data); unparsed counts as wrong",
        "provenance": run_provenance,
        "out": str(out_path),
    }
    if args.summary:
        p = Path(args.summary)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=2))
    print(f"\n[{args.model_label}] n={total} correct={n_correct} "
          f"strict_acc={summary['strict_accuracy_pct']:.2f}% "
          f"parse_success={summary['parse_success_rate']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
