#!/usr/bin/env python
"""P0 -- environment + processor/tokenizer audit, and the matched-processing freeze.

Spec: 04_CURRENT_900_DETAILED_PROTOCOL.md Stage 0,
      07_IMPLEMENTATION_AND_ACCEPTANCE_TESTS.md §1, §2, §4.

Why this matters here, concretely: the 900 source images are not one size.  ALL 300 MRI
are 224x224 (50176 px) so no pixel cap ever binds on them, but **153 of the 900 images
exceed 401408 pixels** and the X-ray subset runs up to 8.8 MP.  Under Qwen2-VL's own
default (max_pixels ~12.8 MP) such an image keeps ~11k patches, while MedVLM-R1's public
processor (max_pixels=401408) keeps 512.  Any cross-checkpoint internal comparison run
without freezing the processor would therefore be comparing different inputs.  This
script measures that difference rather than asserting it.

Establishes, before any internal comparison is allowed:

  1. tokenizer / chat-template / processor content hashes for both checkpoints;
  2. the SAME image yields the SAME number of visual tokens in both checkpoints under
     the frozen matched processor (spec acceptance test);
  3. genuine no-image input yields ZERO visual tokens (no blank-image substitute);
  4. every candidate answer letter is a single token in both tokenizers, so direct
     next-token normalisation is legitimate (spec §4).

Writes provenance/processor_audit.json and provenance/PROCESSOR_AUDIT.md

    python scripts/m1_processor_audit.py [--gpu 3]
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v2 as V   # noqa: E402

# MRI and CT audit items are 224x224 (caps do not bind); Xray_002 is 8.8 MP (cap binds)
AUDIT_SAMPLES = ["MRI_000", "CT_000", "Xray_002"]
FILES_TO_HASH = ["config.json", "preprocessor_config.json", "tokenizer_config.json",
                 "generation_config.json", "chat_template.json", "processor_config.json"]


def sha256_or_none(p: Path) -> str | None:
    return V.sha256_file(p) if p.is_file() else None


def dir_hash(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: x.name):
        if p.is_file():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def model_file_hashes(mdir: Path) -> dict:
    out = {f: sha256_or_none(mdir / f) for f in FILES_TO_HASH}
    tok_files = [p for p in mdir.glob("*.json")
                 if any(k in p.name for k in ("vocab", "merges", "tokenizer"))]
    tok_files += [p for p in mdir.glob("*.txt") if "vocab" in p.name]
    out["tokenizer_content_sha256"] = dir_hash(tok_files)
    out["tokenizer_files"] = sorted(p.name for p in tok_files)
    out["weight_files"] = sorted((p.name, p.stat().st_size)
                                 for p in mdir.glob("*.safetensors"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    args = ap.parse_args()

    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor

    V.ledger_upsert("current900_audit_matched_processor", priority=0,
                    dataset="official900", model="both", condition="audit",
                    status="running", start_time=V.now(),
                    config_path="configs/current900_probe_tasks.yaml")

    items = V.load_items()
    imgs, audit_imgs = {}, {}
    for sid in AUDIT_SAMPLES:
        p = V.source_image_path(items[sid])
        im = Image.open(p).convert("RGB")
        imgs[sid] = im
        audit_imgs[sid] = {
            "path": str(p), "sha256": sha256_or_none(p), "size_wh": list(im.size),
            "pixels": im.size[0] * im.size[1],
            "exceeds_matched_max_pixels": im.size[0] * im.size[1] > V.MATCHED_MAX_PIXELS,
        }

    audit: dict = {
        "created": V.now(),
        "spec_bundle": "medical_vlm_overnight_deepseek_bundle",
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count(),
            "nvidia_smi": V.gpu_snapshot(),
            "selected_gpu": args.gpu,
        },
        "matched_processor": {
            "min_pixels": V.MATCHED_MIN_PIXELS,
            "max_pixels": V.MATCHED_MAX_PIXELS,
            "frozen_for": ["qwen2vl", "medvlmr1"],
            "rationale": ("MedVLM-R1's public preprocessor uses max_pixels=401408 while "
                          "the base Qwen checkpoint defaults far higher; the spec freezes "
                          "the MedVLM-R1 budget for both so cross-checkpoint internal "
                          "comparisons are not confounded by image resolution."),
        },
        "audit_images": audit_imgs,
        "models": {},
    }

    # how many of the 900 actually hit the cap
    from PIL import Image as _I
    n_over = 0
    for it in items.values():
        p = V.source_image_path(it)
        if p.is_file():
            w, h = _I.open(p).size
            if w * h > V.MATCHED_MAX_PIXELS:
                n_over += 1
    audit["dataset_image_budget"] = {
        "n_items": len(items), "n_images_exceeding_matched_max_pixels": n_over,
        "fraction": n_over / len(items),
    }

    for label, mdir in V.MODELS.items():
        m: dict = {"path": str(mdir), "hf_id": V.MODEL_HF_ID[label],
                   "exists": mdir.is_dir()}
        if not mdir.is_dir():
            m["error"] = "missing"
            audit["models"][label] = m
            continue
        m["files"] = model_file_hashes(mdir)

        proc = AutoProcessor.from_pretrained(str(mdir))
        ip = proc.image_processor

        def px(attr):
            return getattr(ip, attr, None)

        m["native_processor"] = {
            "class": type(ip).__name__,
            "min_pixels": px("min_pixels"), "max_pixels": px("max_pixels"),
            "size": (dict(ip.size) if isinstance(getattr(ip, "size", None), dict)
                     else getattr(ip, "size", None)),
            "patch_size": getattr(ip, "patch_size", None),
            "merge_size": getattr(ip, "merge_size", None),
            "temporal_patch_size": getattr(ip, "temporal_patch_size", None),
        }

        tok = proc.tokenizer
        m["tokenizer"] = {
            "class": type(tok).__name__,
            "vocab_size": getattr(tok, "vocab_size", None),
            "len": len(tok),
            "additional_special_tokens": [str(t) for t in
                                          getattr(tok, "additional_special_tokens", [])],
            "chat_template_sha256": hashlib.sha256(
                (tok.chat_template or "").encode()).hexdigest(),
        }
        for name in ("image_token", "video_token", "vision_start_token",
                     "vision_end_token", "pad_token", "eos_token"):
            tk = getattr(proc, name, None) or getattr(tok, name, None)
            tid = None
            if tk is not None:
                try:
                    tid = tok.convert_tokens_to_ids(tk)
                except Exception:
                    tid = None
            m["tokenizer"][f"{name}_id"] = tid

        m["candidate_tokenization"] = {
            L: {"ids": tok.encode(L, add_special_tokens=False),
                "n_tokens": len(tok.encode(L, add_special_tokens=False)),
                "piece": tok.convert_ids_to_tokens(
                    tok.encode(L, add_special_tokens=False))}
            for L in "ABCDZ"}
        m["all_candidates_single_token"] = all(
            v["n_tokens"] == 1 for v in m["candidate_tokenization"].values())
        img_tok = m["tokenizer"].get("image_token_id")

        def count_visual(processor, text_prompt, image) -> dict:
            """image=None is GENUINE no-image: no image object is passed at all."""
            content = ([{"type": "image", "image": image}] if image is not None else [])
            content.append({"type": "text", "text": text_prompt})
            msgs = [{"role": "user", "content": content}]
            text = processor.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True)
            if image is not None:
                from qwen_vl_utils import process_vision_info
                im, vid = process_vision_info(msgs)
                enc = processor(text=[text], images=im, videos=vid,
                                return_tensors="pt", padding=True)
            else:
                enc = processor(text=[text], return_tensors="pt", padding=True)
            ids = enc["input_ids"][0].tolist()
            return {"seq_len": len(ids), "n_visual_tokens": ids.count(img_tok),
                    "has_pixel_values": "pixel_values" in enc,
                    "grid_thw": (enc["image_grid_thw"].tolist()
                                 if "image_grid_thw" in enc else None),
                    "pixel_values_shape": (list(enc["pixel_values"].shape)
                                           if "pixel_values" in enc else None),
                    "input_ids_head": ids[:8], "input_ids_tail": ids[-6:]}

        m["visual_tokens"] = {}
        for sid in AUDIT_SAMPLES:
            ordered = V.apply_order(items[sid], "O0")
            prompt = V.build_m2_prompt(ordered["mcq_text"])
            m["visual_tokens"][sid] = {}
            # native processor
            m["visual_tokens"][sid]["native_clean"] = count_visual(proc, prompt,
                                                                   imgs[sid])
            m["visual_tokens"][sid]["native_no_image"] = count_visual(proc, prompt, None)
            # matched processor
            ip.min_pixels = V.MATCHED_MIN_PIXELS
            ip.max_pixels = V.MATCHED_MAX_PIXELS
            m["visual_tokens"][sid]["matched_clean"] = count_visual(proc, prompt,
                                                                    imgs[sid])
            m["visual_tokens"][sid]["matched_no_image"] = count_visual(proc, prompt,
                                                                       None)
            ip.min_pixels = m["native_processor"]["min_pixels"]
            ip.max_pixels = m["native_processor"]["max_pixels"]

        m["matched_processor_applied"] = {"min_pixels": V.MATCHED_MIN_PIXELS,
                                          "max_pixels": V.MATCHED_MAX_PIXELS}
        m["no_image_zero_visual_tokens"] = all(
            m["visual_tokens"][s]["matched_no_image"]["n_visual_tokens"] == 0
            for s in AUDIT_SAMPLES)
        audit["models"][label] = m
        del proc
        gc.collect()

    # ---- acceptance ---------------------------------------------------------- #
    q, d = audit["models"].get("qwen2vl", {}), audit["models"].get("medvlmr1", {})
    eq = {s: (q.get("visual_tokens", {}).get(s, {}).get("matched_clean", {})
              .get("n_visual_tokens")
              == d.get("visual_tokens", {}).get(s, {}).get("matched_clean", {})
              .get("n_visual_tokens")) for s in AUDIT_SAMPLES}
    audit["acceptance"] = {
        "matched_visual_token_count_equal_per_image": eq,
        "matched_visual_token_count_equal": all(eq.values()),
        "native_visual_token_count_qwen": {
            s: q.get("visual_tokens", {}).get(s, {}).get("native_clean", {})
            .get("n_visual_tokens") for s in AUDIT_SAMPLES},
        "native_visual_token_count_medvlmr1": {
            s: d.get("visual_tokens", {}).get(s, {}).get("native_clean", {})
            .get("n_visual_tokens") for s in AUDIT_SAMPLES},
        "no_image_zero_visual_tokens_qwen": q.get("no_image_zero_visual_tokens"),
        "no_image_zero_visual_tokens_medvlmr1": d.get("no_image_zero_visual_tokens"),
        "all_candidates_single_token_qwen": q.get("all_candidates_single_token"),
        "all_candidates_single_token_medvlmr1": d.get("all_candidates_single_token"),
    }
    audit["acceptance"]["PASS"] = all([
        audit["acceptance"]["matched_visual_token_count_equal"],
        audit["acceptance"]["no_image_zero_visual_tokens_qwen"] is True,
        audit["acceptance"]["no_image_zero_visual_tokens_medvlmr1"] is True,
        audit["acceptance"]["all_candidates_single_token_qwen"] is True,
        audit["acceptance"]["all_candidates_single_token_medvlmr1"] is True,
    ])

    V.write_json(V.ROOT / "provenance" / "processor_audit.json", audit)

    # ---- markdown ------------------------------------------------------------ #
    acc = audit["acceptance"]
    L = ["# Processor / Tokenizer Audit (P0)", "",
         f"generated {audit['created']} · spec bundle "
         "`medical_vlm_overnight_deepseek_bundle`", "",
         "## Environment", "",
         f"- python {audit['environment']['python']}, torch "
         f"{audit['environment']['torch']}, transformers "
         f"{audit['environment']['transformers']}",
         f"- cuda available: {audit['environment']['cuda_available']}, GPUs visible: "
         f"{audit['environment']['gpu_count']}", "",
         "## Why the matched processor is not a formality", "",
         f"- **{n_over} of {len(items)} "
         f"({100 * n_over / len(items):.1f}%)** source images exceed "
         f"`max_pixels={V.MATCHED_MAX_PIXELS}`.",
         "- all 300 MRI images are a uniform 224x224, so the cap never binds on MRI — "
         "which is exactly why an MRI-only audit would have missed this.",
         "- the X-ray subset runs up to 8.8 MP, where the native Qwen budget and the "
         "MedVLM-R1 budget differ by more than an order of magnitude in visual tokens.",
         "",
         "Full procedure — `min_pixels = "
         f"{V.MATCHED_MIN_PIXELS}`, `max_pixels = {V.MATCHED_MAX_PIXELS}` **for both "
         "checkpoints**:", "",
         "| audit item | pixels | native Qwen visual tokens | native MedVLM-R1 visual "
         "tokens | matched (both) |",
         "|---|---|---|---|---|"]
    for s in AUDIT_SAMPLES:
        a = audit_imgs[s]
        matched_n = (q.get("visual_tokens", {}).get(s, {})
                     .get("matched_clean", {}).get("n_visual_tokens"))
        L.append(f"| `{s}` | {a['pixels']:,} | "
                 f"{acc['native_visual_token_count_qwen'].get(s)} | "
                 f"{acc['native_visual_token_count_medvlmr1'].get(s)} | "
                 f"{matched_n} |")

    L += ["", "## Per-checkpoint facts", "",
          "| item | Qwen2-VL-2B-Instruct | MedVLM-R1 | same? |", "|---|---|---|---|"]
    rows = [
        ("config.json sha256", lambda m: (m["files"]["config.json"] or "")[:12] + "…"),
        ("preprocessor_config.json sha256",
         lambda m: (m["files"]["preprocessor_config.json"] or "")[:12] + "…"),
        ("chat template sha256",
         lambda m: (m["tokenizer"]["chat_template_sha256"] or "")[:12] + "…"),
        ("tokenizer content sha256",
         lambda m: (m["files"]["tokenizer_content_sha256"] or "")[:12] + "…"),
        ("tokenizer vocab size", lambda m: m["tokenizer"]["vocab_size"]),
        ("native min_pixels", lambda m: m["native_processor"]["min_pixels"]),
        ("native max_pixels", lambda m: m["native_processor"]["max_pixels"]),
        ("patch / merge size",
         lambda m: f"{m['native_processor']['patch_size']} / "
                   f"{m['native_processor']['merge_size']}"),
        ("image_token_id", lambda m: m["tokenizer"]["image_token_id"]),
        ("all A–D,Z single tokens", lambda m: m["all_candidates_single_token"]),
        ("A/Z token ids",
         lambda m: f"{m['candidate_tokenization']['A']['ids']} / "
                   f"{m['candidate_tokenization']['Z']['ids']}"),
        ("seq len, matched MRI_000 clean",
         lambda m: m["visual_tokens"]["MRI_000"]["matched_clean"]["seq_len"]),
        ("visual tokens, matched no_image",
         lambda m: m["visual_tokens"]["MRI_000"]["matched_no_image"]["n_visual_tokens"]),
    ]
    for name, fn in rows:
        try:
            a, b = fn(q), fn(d)
        except Exception as exc:
            a = b = f"n/a ({exc})"
        L.append(f"| {name} | {a} | {b} | {'**yes**' if a == b else 'no'} |")

    verdict = ("PASS" if acc["PASS"]
               else "FAIL - DO NOT PROCEED TO INTERNAL COMPARISONS")
    L += ["", "## Acceptance tests (spec §1, §2, §4)", "",
          f"- matched visual token count equal across checkpoints, per audit image: "
          f"**{acc['matched_visual_token_count_equal']}** — {eq}",
          f"- genuine no-image yields zero visual tokens: qwen "
          f"{acc['no_image_zero_visual_tokens_qwen']}, MedVLM-R1 "
          f"{acc['no_image_zero_visual_tokens_medvlmr1']}",
          f"- every candidate label is a single token: qwen "
          f"{acc['all_candidates_single_token_qwen']}, MedVLM-R1 "
          f"{acc['all_candidates_single_token_medvlmr1']}",
          f"- **overall: {verdict}**", "",
          "Machine-readable: `provenance/processor_audit.json`.", ""]
    (V.ROOT / "provenance" / "PROCESSOR_AUDIT.md").write_text("\n".join(L),
                                                             encoding="utf-8")

    V.ledger_upsert("current900_audit_matched_processor",
                    status=("complete" if acc["PASS"] else "partial"),
                    end_time=V.now(), output_path="provenance/processor_audit.json",
                    blocker_or_error="" if acc["PASS"] else "acceptance test failed")

    print("native vs matched visual tokens")
    for s in AUDIT_SAMPLES:
        print(f"  {s:9s} px={audit_imgs[s]['pixels']:>9,}  "
              f"qwen_native={acc['native_visual_token_count_qwen'].get(s)!s:>6}  "
              f"med_native={acc['native_visual_token_count_medvlmr1'].get(s)!s:>6}  "
              f"matched={q['visual_tokens'][s]['matched_clean']['n_visual_tokens']}")
    print(f"images exceeding matched max_pixels: {n_over}/{len(items)}")
    print(f"no-image zero visual tokens: "
          f"{acc['no_image_zero_visual_tokens_qwen']} / "
          f"{acc['no_image_zero_visual_tokens_medvlmr1']}")
    print(f"single-token candidates: {acc['all_candidates_single_token_qwen']} / "
          f"{acc['all_candidates_single_token_medvlmr1']}")
    print(f"ACCEPTANCE: {'PASS' if acc['PASS'] else 'FAIL'}")
    return 0 if acc["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
