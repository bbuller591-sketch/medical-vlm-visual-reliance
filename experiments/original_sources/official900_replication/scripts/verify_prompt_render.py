#!/usr/bin/env python
"""Verify the rendered prompt byte-for-byte against the official script's logic.

Prompt construction is the single most likely place for a silent replication error,
so it is checked independently rather than assumed: this builds the message exactly
the way ``src/eval/test_qwen2vl_med.py`` does - the same two QUESTION_TEMPLATE
strings, the same single user turn, the same ``apply_chat_template`` arguments - and
diffs the result against the prompt our runner actually used (dumped by
``--dump-prompts``).

It deliberately does NOT import anything from our runner, so an error shared between
the two implementations cannot hide.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Copied from the official file, including the trailing spaces on the first line
# and the four-space indent.
TEMPLATES = {
    "complex": """
    {Question} 
    Your task: 
    1. Think through the question step by step, enclose your reasoning process in <think>...</think> tags. 
    2. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags.
    3. No extra information or text outside of these tags.
    """,
    "simple": """
    {Question} 
    Your task: provide the correct single-letter choice (A, B, C, D,...).
    """,
}


def main() -> int:
    from transformers import AutoProcessor

    subset = json.loads((ROOT / "diagnostics" / "official_crosscheck_subset.json").read_text())
    base = ROOT / "data" / "images_extracted"
    out_dir = ROOT / "diagnostics" / "rendered_prompts"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {}
    for label, model, prompt_type, jsonl in (
        ("qwen2vl_cc", "<LOCAL_PROJECT_ROOT>/medical/models/Qwen2-VL-2B-Instruct",
         "simple", ROOT / "diagnostics" / "ours_qwen_20.jsonl"),
        ("medvlmr1_cc", "<LOCAL_PROJECT_ROOT>/medical/models/MedVLM-R1",
         "complex", ROOT / "diagnostics" / "ours_medvlm_20.jsonl"),
    ):
        proc = AutoProcessor.from_pretrained(model)
        target = TEMPLATES[prompt_type]
        # Map image path -> the sample_id our runner used. The subset file has 20
        # rows, so the runner's positional modality slicing collapses everything into
        # the first block; matching on the dumped filename alone would silently
        # compare only a seventh of the subset, which is exactly what it did first.
        sid_by_image = {}
        for line in jsonl.open():
            if line.strip():
                r = json.loads(line)
                sid_by_image[r["image"]] = r["sample_id"]

        n_ok, n_cmp, diffs = 0, 0, []
        for row in subset:
            sid = sid_by_image.get(row["image"][0])
            if sid is None:
                continue
            # exactly the official construction
            message = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{base}/{row['image'][0]}"},
                    {"type": "text", "text": target.format(Question=row["problem"])},
                ],
            }]
            text = proc.apply_chat_template(message, tokenize=False,
                                            add_generation_prompt=True)

            fp = out_dir / f"{label}_{sid}.txt"
            if not fp.is_file():
                continue
            n_cmp += 1
            ours = fp.read_text()
            if ours == text:
                n_ok += 1
            else:
                diffs.append({"sample_id": sid,
                              "official_len": len(text), "ours_len": len(ours),
                              "first_diff_at": next(
                                  (i for i, (a, b) in enumerate(zip(text, ours))
                                   if a != b), min(len(text), len(ours))),
                              "official_head": text[:200], "ours_head": ours[:200]})
        print(f"  {label:<14} prompt identical: {n_ok}/{n_cmp}")
        for d in diffs[:3]:
            print(f"     DIFF {json.dumps(d, ensure_ascii=False)[:300]}")
        result[label] = {"n_compared": n_cmp, "n_identical": n_ok, "diffs": diffs[:5]}

    p = ROOT / "results" / "prompt_crosscheck.json"
    p.write_text(json.dumps(result, indent=2))
    print(f"[prompt-check] -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
