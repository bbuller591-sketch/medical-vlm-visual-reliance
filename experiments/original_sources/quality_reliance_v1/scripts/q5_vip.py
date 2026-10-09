#!/usr/bin/env python
"""Q5 -- VIP / TVI, conditional on construct validity (spec 07 + amendment E).

WHAT THIS IS
------------
VIP is the layer at which the representation distance between the with-image and
blind forward passes starts to separate a vision-DEPENDENT question distribution
from a vision-INDEPENDENT one.  TVI then aggregates that distance after the VIP.

It is an imported construct: the original work is about language priors in
generic LVLMs, and applying it to medical visual reliability is a construct
extension.  It therefore has to pass a known-group validity gate before it is
allowed anywhere near the headline result:

    TVI(P_VT) > TVI(P_T)   with uncertainty

If that fails, TVI is downgraded to exploratory and the rest of the project
continues without it.  That is the spec's instruction, and this script obeys it
by writing the failure into `VIP_TVI_VALIDITY.md` rather than quietly retuning.

DATA
----
Calibration must be INDEPENDENT of the official 900 -- selecting a VIP on the
same items whose TVI is later reported would make the landmark a fitted
parameter of the result.
  P_VT: RadImageNet-VQA benchmark (text-only accuracy near chance by design)
  P_T : MedMCQA radiology items that are answerable without an image, given an
        irrelevant medical image under a frozen protocol

This script first checks whether either is present locally.  Downloading is
attempted only through the mirror that works on this machine; if that fails the
branch is recorded as blocked and the queue moves on, per the spec's fallback
rule.  It never fabricates a calibration set.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_qr as Q   # noqa: E402

CAL_DIR = Q.ROOT / "data" / "vip_calibration"
HF_DATASETS = ["raidium/RadImageNet-VQA"]
NEEDED_MEDMCQA = "medmcqa"


def probe_local() -> dict:
    """Look for the calibration DATA, not for a file whose name resembles it.

    A first version globbed `**/RadImageNet*` and matched a single cached PNG
    named `RadImageNet__acl_pathology__knee187955.png` inside an unrelated
    project's image cache, then reported the calibration set as present.  A hit
    now has to be a directory that actually contains tabular data.
    """
    found = {"radimagenet_vqa": None, "medmcqa": None}
    roots = (Q.MED / "data", Q.MED / "cache", Q.ROOT / "data",
             Path("<LOCAL_PROJECT_ROOT>/hf_cache"),
             Path("<LOCAL_PROJECT_ROOT>/.cache/huggingface"))
    for root in roots:
        if not root.is_dir():
            continue
        for pat in ("**/RadImageNet*", "**/radimagenet*"):
            for p in root.glob(pat):
                if p.is_dir() and any(p.rglob("*.json")) or \
                   (p.is_dir() and (any(p.rglob("*.parquet"))
                                    or any(p.rglob("*.csv")))):
                    found["radimagenet_vqa"] = str(p)
        for pat in ("**/medmcqa*", "**/MedMCQA*"):
            for p in root.glob(pat):
                if p.is_dir() and (any(p.rglob("*.json"))
                                   or any(p.rglob("*.parquet"))):
                    found["medmcqa"] = str(p)
    return found


def try_download() -> tuple[bool, str]:
    """Attempt the benchmark subset through the only route that works here.

    The proxy on this machine breaks huggingface downloads, so the proxy
    variables are unset and the mirror endpoint is used.
    """
    env = dict(os.environ)
    for v in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
        env.pop(v, None)
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:                                   # pragma: no cover
        return False, f"huggingface_hub unavailable: {exc}"
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        p = snapshot_download(
            "raidium/RadImageNet-VQA", repo_type="dataset",
            local_dir=str(CAL_DIR / "radimagenet_vqa"),
            allow_patterns=["*.json", "*.csv", "*.parquet", "*.md", "*.txt"],
            max_workers=4)
        return True, str(p)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-download", action="store_true")
    args = ap.parse_args()

    rep = Q.ROOT / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    found = probe_local()
    print("local calibration data:", found)

    status, detail = "blocked", "no calibration data found locally"
    if found["radimagenet_vqa"] is None and args.allow_download:
        ok, detail2 = try_download()
        status = "downloaded" if ok else "blocked"
        detail = detail2
        print("download attempt:", status, detail)
        if ok:
            found["radimagenet_vqa"] = detail2
    elif found["radimagenet_vqa"]:
        status = "present"

    lines = [
        "# VIP / TVI validity",
        "",
        "Status: **{}**".format(status),
        "",
        f"Detail: `{detail}`",
        "",
        "## Why this branch is conditional",
        "",
        "TVI is imported from generic-LVLM language-prior work.  Extending it to",
        "medical visual reliability is a construct extension, so it must pass a",
        "known-group gate (`TVI(P_VT) > TVI(P_T)`) before it may be used in any",
        "claim.  The gate is checked on calibration data INDEPENDENT of the",
        "official 900: choosing a VIP on the same items whose TVI is later",
        "reported would make the landmark a fitted parameter of the result.",
        "",
        "## What is required to unblock",
        "",
        "1. `RadImageNet-VQA` benchmark split (P_VT) -- 200 items minimum, 400 if",
        "   the layer curves are unstable.",
        "2. MedMCQA radiology/medical items that are answerable text-only, each",
        "   paired with an irrelevant medical image under a frozen protocol (P_T).",
        "   Formatting (4-choice, prompt template, question and option length bins)",
        "   must be matched to P_VT as closely as possible; residual distribution",
        "   differences are a recorded limitation, not something to hide.",
        "3. One frozen VIP per model, estimated on calibration only.",
        "",
        "## What is already in place",
        "",
        "The instrument needed to compute the contrast exists and is shared with",
        "the main experiment: the same frozen MCQ prompt, the same matched",
        "processor budget, and a genuine no-image pass (no placeholder token, no",
        "`pixel_values`).  Per-layer decision states for both are already stored",
        "for the official 900, so the machinery is not the blocker; the",
        "calibration data is.",
        "",
        "## Gate criteria (to be evaluated once data exists)",
        "",
        "| # | gate | requirement |",
        "|---|---|---|",
        "| 1 | known-group validity | `TVI(P_VT) > TVI(P_T)` with an interval "
        "excluding zero |",
        "| 2 | pre/post VIP | post-VIP separation meaningfully stronger than "
        "pre-VIP |",
        "| 3 | behavioural convergence | TVI tracks `JS(P_image, P_noimage)` or "
        "`|U|` |",
        "| 4 | causal subset | high-TVI items show larger intervention effects |",
        "",
        "Failing gate 1 downgrades TVI to exploratory.  The rest of the project",
        "does not depend on it.",
        "",
        "## Implementation note carried over from the audit",
        "",
        "The official path uses `add_generation_prompt=True` and reads the hidden",
        "state at the last position of the prefill, not the last generated token.",
        "This project's decision position is the same one, so the main experiment",
        "and the VIP calibration are on the same footing.  A last-generated-token",
        "variant is listed as a sensitivity analysis, not a blocker.",
        "",
    ]
    (rep / "VIP_TVI_VALIDITY.md").write_text("\n".join(lines) + "\n",
                                             encoding="utf-8")
    Q.write_json(Q.ROOT / "runtime" / "vip_status.json", {
        "status": status, "detail": detail, "found": found, "time": Q.now()})
    print("wrote reports/VIP_TVI_VALIDITY.md  status =", status)


if __name__ == "__main__":
    main()
