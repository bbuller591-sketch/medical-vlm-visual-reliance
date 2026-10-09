#!/usr/bin/env python
"""Shared library for visual_reliance_mechanism_v2.

The overnight spec (medical_vlm_overnight_deepseek_bundle) requires that the old
condition construction and design constants be REUSED rather than rewritten, so that
the mechanism run is comparable to the behavioural run.  This module therefore imports
the frozen design out of the previous project instead of copying it:

    official900_visual_reliance_v1/scripts/common.py

That import is read-only.  Nothing in this project ever writes into the old project.

WHAT IS NEW HERE (and only here):
  * the matched image-processing rule (min_pixels/max_pixels) that the overnight spec
    mandates for BOTH checkpoints, and
  * the token-level plumbing needed for hidden-state extraction.

FROZEN, from the spec (do not change without recording why in provenance/):
    MATCHED_MIN_PIXELS = 3136
    MATCHED_MAX_PIXELS = 401408
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]                 # visual_reliance_mechanism_v2
MED = ROOT.parent                                          # <LOCAL_PROJECT_ROOT>/medical
OLD = MED / "official900_visual_reliance_v1"               # read-only source of truth

sys.path.insert(0, str(OLD / "scripts"))
import common as OD   # noqa: E402  frozen design: conditions, option orders, params

for _d in ("configs", "provenance", "manifests", "scripts", "status", "logs",
           "features", "probes", "causal", "plots", "reports"):
    (ROOT / _d).mkdir(parents=True, exist_ok=True)

LEDGER = ROOT / "status" / "experiment_ledger.csv"
LEDGER_COLS = ["experiment_id", "priority", "dataset", "model", "condition",
               "sample_count_planned", "sample_count_done", "gpu_id", "status",
               "start_time", "end_time", "config_path", "output_path",
               "blocker_or_error"]

# --------------------------------------------------------------------------- #
# Frozen design (re-exported so stage scripts import one module)
# --------------------------------------------------------------------------- #
GLOBAL_SEED = 20260916
OPTION_ORDERS = OD.OPTION_ORDERS
ABSTAIN_LETTER = OD.ABSTAIN_LETTER
apply_order = OD.apply_order
build_m2_prompt = OD.build_m2_prompt
stable_seed = OD.stable_seed
# re-exported so callers never have to reach through `OD` for the two functions that
# together define how an item is rendered; using one without the other silently
# desynchronises the prompt from the logit->semantic inversion.
option_order_perm = OD.option_order_perm
build_mcq_text = OD.build_mcq_text

# the O0 mechanism conditions, in spec order.  `balanced_mismatch` is the P0-B
# replacement for the parent-pool donor condition: its image is the clean image of a
# derangement partner, so the two classes share one image multiset exactly.
MECH_CONDITIONS = ["clean", "no_image", "blur_severe", "modality_specific_severe",
                   "random_same_modality_replacement", "same_answer_donor",
                   "opposite_answer_donor", "balanced_mismatch"]
# conditions that need a donor row in donor_maps.json
DONOR_CONDITIONS = {"random_same_modality_replacement", "same_answer_donor",
                    "opposite_answer_donor", "ood_cross_modality"}

# --------------------------------------------------------------------------- #
# Matched image processing -- THE central P0 control
# --------------------------------------------------------------------------- #
MATCHED_MIN_PIXELS = 3136
MATCHED_MAX_PIXELS = 401408
MATCHED_PROCESSOR_ID = f"matched_min{MATCHED_MIN_PIXELS}_max{MATCHED_MAX_PIXELS}"

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
MODELS = {
    "qwen2vl": MED / "models" / "Qwen2-VL-2B-Instruct",
    "medvlmr1": MED / "models" / "MedVLM-R1",
}
MODEL_DISPLAY = {"qwen2vl": "Qwen2-VL-2B-Instruct", "medvlmr1": "MedVLM-R1"}
MODEL_HF_ID = {"qwen2vl": "Qwen/Qwen2-VL-2B-Instruct", "medvlmr1": "JZPeterPan/MedVLM-R1"}

# Qwen2-VL special ids we depend on; verified at runtime by m1_processor_audit.py
IMAGE_TOKEN_CANDIDATES = (151655,)

# --------------------------------------------------------------------------- #
# Time / io helpers
# --------------------------------------------------------------------------- #
def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(p) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_json(p, obj) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def load_jsonl(p) -> list:
    p = Path(p)
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def append_jsonl(p, rec: dict) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def dedupe_load_glob(directory: Path, patterns) -> list:
    """Union glob patterns BY FILE NAME before reading.

    The old project hit a real bug where `M2_*.jsonl` also matched `M2__*.jsonl`, so
    every file was read twice.  Harmless for dict-keyed means, fatal for counts.  Kept
    as a helper here so no stage can reintroduce it.
    """
    files = sorted({p for pat in patterns for p in Path(directory).glob(pat)})
    out = []
    for p in files:
        out += load_jsonl(p)
    return out


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #
def ledger_upsert(experiment_id: str, **fields) -> None:
    """Append-or-update one row of status/experiment_ledger.csv.

    Resumability depends on this file surviving a crash, so it is rewritten
    atomically after every update rather than held in memory.
    """
    rows = []
    if LEDGER.is_file():
        with open(LEDGER, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    for r in rows:
        if r.get("experiment_id") == experiment_id:
            r.update({k: ("" if v is None else str(v)) for k, v in fields.items()})
            break
    else:
        row = {c: "" for c in LEDGER_COLS}
        row["experiment_id"] = experiment_id
        row.update({k: ("" if v is None else str(v)) for k, v in fields.items()})
        rows.append(row)
    tmp = LEDGER.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LEDGER_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in LEDGER_COLS})
    tmp.replace(LEDGER)


# --------------------------------------------------------------------------- #
# Dataset / conditions
# --------------------------------------------------------------------------- #
def load_items() -> dict:
    """The 900 official items, reused verbatim from the old manifest (cached)."""
    global _ITEMS
    if _ITEMS is None:
        man = read_json(OLD / "manifests" / "dataset_manifest.json")
        _ITEMS = {it["sample_id"]: it for it in man["items"]}
    return _ITEMS


_ITEMS: dict | None = None
_DERANG: dict | None = None


def load_derangement() -> dict:
    """P0-B balanced mismatch assignment (built by scripts/p0_derangement.py)."""
    global _DERANG
    if _DERANG is None:
        p = ROOT / "manifests" / "balanced_mismatch.json"
        _DERANG = read_json(p) if p.is_file() else {}
    return _DERANG


def load_option_orders() -> dict:
    return read_json(OLD / "manifests" / "option_orders.json")


def load_donors() -> dict:
    return read_json(OLD / "manifests" / "donor_maps.json")


def donor_entry(donors: dict, sample_id: str, condition: str) -> dict | None:
    e = donors.get(sample_id, {}).get(condition)
    if not isinstance(e, dict) or "donor_image_sha256" not in e:
        return None
    return e


def conditions_for(donors: dict, sample_id: str) -> list:
    """The O0 mechanism conditions actually available for this sample."""
    out = []
    for c in MECH_CONDITIONS:
        if c in DONOR_CONDITIONS and donor_entry(donors, sample_id, c) is None:
            continue
        out.append(c)
    return out


def source_image_path(item: dict) -> Path:
    """The clean image.  Old project read these from the replication extraction."""
    return (MED / "official900_replication" / "data" / "images_extracted" / "images"
            / item["image_file"])


def image_path(condition: str, item: dict) -> Path | None:
    """None means GENUINE no-image: no image object is passed at all.

    A blank/white image is a different control and is never substituted here.

    `balanced_mismatch` returns the clean image of the item's derangement partner --
    a benchmark image, not a parent-pool donor.  That is what makes the two probe
    classes share one image multiset.
    """
    if condition == "no_image":
        return None
    if condition == "clean":
        return source_image_path(item)
    if condition == "balanced_mismatch":
        d = load_derangement().get("assignment", {}).get(item["sample_id"])
        if not d:
            return None
        return source_image_path(load_items()[d["donor_sample_id"]])
    return OLD / "generated_images" / condition / f"{item['sample_id']}.png"


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def sha256_file(p: Path) -> str:
    return OD.sha256_file(Path(p))


def gpu_snapshot() -> list:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader"], capture_output=True, text=True, check=True)
        return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    except Exception as exc:                                    # pragma: no cover
        return [f"nvidia-smi failed: {exc}"]


def pick_idle_gpus(max_n: int = 2, need_free_mib: int = 20000) -> list[int]:
    """Lowest-index GPUs that are idle AND have room.  Never touches other jobs."""
    import re
    free = []
    for ln in gpu_snapshot():
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 5:
            continue
        idx = int(parts[0])
        used = int(re.sub(r"\D", "", parts[2]) or 0)
        total = int(re.sub(r"\D", "", parts[3]) or 0)
        util = int(re.sub(r"\D", "", parts[4]) or 0)
        if total - used >= need_free_mib and util <= 5:
            free.append((idx, total - used))
    free.sort()
    return [i for i, _ in free[:max_n]]
