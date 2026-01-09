#!/usr/bin/env python3
"""
Batch inference script for ConTEXTual_Net_3D over a JSON of label texts.

For each JSON entry like:
  PETWB_000001_04_label_1: "<report snippet>"

We:
  - parse CASE_ID = PETWB_000001_04
  - load:
      data/<CASE_ID>/<CASE_ID>_ct_cropped.nii.gz
      data/<CASE_ID>/<CASE_ID>_pet_cropped.nii.gz   (or _suv_cropped.nii.gz if you prefer)
  - run the exact same inference pipeline you provided
  - save per-label masks:
      data/<CASE_ID>/mask1.nii.gz, mask2.nii.gz, ...
    If only one label exists for that case, we still save mask1.nii.gz.

All config is via globals below (no CLI args).
"""

import os
import re
import json
from pathlib import Path

import numpy as np
import torch
import nibabel as nib
from transformers import AutoTokenizer, RobertaModel

from cont import ConTEXTual_Net_3D  # ensure on PYTHONPATH


# -------------------------------------------------------------------------
# USER-CONTROLLED GLOBALS
# -------------------------------------------------------------------------

# JSON: either provide the dict directly, OR point to a .json file.
# Option A: JSON_PATH to a file:
JSON_PATH = "data.json"  # e.g. "/path/to/labels.json"

# Option B: paste dict here (if JSON_PATH is None, we use LABEL_JSON_DICT)
LABEL_JSON_DICT = None  # set to your dict object

# Data root holding PETWB_*/ subdirs
DATA_ROOT = "/mnt/DGXUserData/dxm060/Danyal/snmmi26/contextual/data"

# Model checkpoint
CHECKPOINT_PATH = "/mnt/DGXUserData/dxm060/Zach_Analysis/final_3d_models_used_in_paper/data_size_ablation/final_1.0_data_for_paper_contextual_net/model_final.pt"

# Roberta directory / name (match training)
#BERT_MODEL_DIR = "FacebookAI/roberta-base"
BERT_MODEL_DIR = "StanfordAIMI/RadBERT"

# Image filename suffixes (choose the one you actually have)
CT_SUFFIX = "_ct_cropped.nii.gz"
# PET_SUFFIX = "_pet_cropped.nii.gz"
PET_SUFFIX = "_suv_cropped.nii.gz"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tokenization settings (match training)
MAX_TEXT_LEN = 512

# Intensity normalization ranges (match training)
PET_MIN, PET_MAX = 0.0, 10.0
CT_MIN, CT_MAX = -150.0, 250.0

# Segmentation threshold on lesion-channel probability
PRED_THRESHOLD = 0.5
bad_count = 0
good_count = 0

# -------------------------------------------------------------------------
# UTILS: I/O + NORMALIZATION
# -------------------------------------------------------------------------

def load_nii(path: str):
    """Load NIfTI, returning (volume, affine, header)."""
    nii = nib.load(path)
    data = nii.get_fdata().astype(np.float32)
    return data, nii.affine, nii.header


def normalize_volume(volume: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Clip volume to [vmin, vmax] and scale to [0, 1]."""
    vol = np.clip(volume, vmin, vmax)
    denom = (vmax - vmin) if (vmax - vmin) != 0 else 1.0
    vol = (vol - vmin) / denom
    return vol.astype(np.float32)


# -------------------------------------------------------------------------
# MODEL BUILDING (loaded once, reused)
# -------------------------------------------------------------------------

def build_model_and_tokenizer():
    """
    Instantiate tokenizer, language model, and ConTEXTual_Net_3D,
    then load checkpoint weights.
    """
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)
    language_model = RobertaModel.from_pretrained(
        BERT_MODEL_DIR,
        output_hidden_states=True,
    )
    for p in language_model.parameters():
        p.requires_grad = False

    n_classes = 2
    kernels = [[3, 3, 3]] * 5
    strides = [
        [1, 1, 1],
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
    ]
    deep_supr_num = len(strides) - 2

    model = ConTEXTual_Net_3D(
        spatial_dims=3,
        in_channels=2,
        out_channels=n_classes,
        kernel_size=kernels,
        filters=[64, 96, 128, 192, 256],
        strides=strides,
        upsample_kernel_size=strides[1:],
        res_block=True,
        norm_name="instance",
        deep_supervision=False,
        deep_supr_num=deep_supr_num,
        language_model=language_model,
    ).to(DEVICE)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    print("Loading state dict...")
    model.load_state_dict(state_dict)
    model.eval()
    print("Model ready.")
    return model, tokenizer


# -------------------------------------------------------------------------
# INPUT PREPARATION
# -------------------------------------------------------------------------

def prepare_image_tensors(pet_vol: np.ndarray, ct_vol: np.ndarray) -> torch.Tensor:
    """
    Prepare PET and CT volumes for the model.

    Input shapes: (H, W, D) and must match.
    Output shape: (1, 2, H, W, D)
    """
    assert pet_vol.shape == ct_vol.shape, "PET and CT must have the same shape."

    pet_norm = normalize_volume(pet_vol, PET_MIN, PET_MAX)
    ct_norm = normalize_volume(ct_vol, CT_MIN, CT_MAX)

    pet = torch.from_numpy(pet_norm[None, ...]).unsqueeze(0)  # (1,1,H,W,D)
    ct = torch.from_numpy(ct_norm[None, ...]).unsqueeze(0)    # (1,1,H,W,D)

    images = torch.stack((pet, ct), dim=1).squeeze(2)          # (1,2,H,W,D)
    return images.to(DEVICE)


def prepare_text_tensors(tokenizer, text: str):
    """
    Tokenize text into ids, mask, token_type_ids tensors of shape (1, L).
    """
    encoded = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=MAX_TEXT_LEN,
        padding="max_length",
        truncation="longest_first",
        return_token_type_ids=True,
    )

    ids = torch.tensor(encoded["input_ids"], dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(encoded["attention_mask"], dtype=torch.long).unsqueeze(0)
    tti = torch.tensor(encoded["token_type_ids"], dtype=torch.long).unsqueeze(0)

    return ids.to(DEVICE), mask.to(DEVICE), tti.to(DEVICE)


# -------------------------------------------------------------------------
# JSON PARSING + PATH RESOLUTION
# -------------------------------------------------------------------------

_KEY_RE = re.compile(r"^(?P<case>PETWB_\d+_\d+)_label_(?P<label_idx>\d+)$")


def load_label_json() -> dict:
    if JSON_PATH is not None:
        with open(JSON_PATH, "r") as f:
            return json.load(f)
    if LABEL_JSON_DICT is None:
        raise ValueError("Set JSON_PATH or LABEL_JSON_DICT.")
    return LABEL_JSON_DICT


def group_entries_by_case(label_json: dict):
    """
    Returns:
      case_to_entries: dict[case_id] = list[(label_idx_int, key, text)]
    """
    case_to_entries = {}
    for k, text in label_json.items():
        m = _KEY_RE.match(k)
        if not m:
            raise ValueError(f"Key does not match expected pattern: {k}")
        case_id = m.group("case")
        label_idx = int(m.group("label_idx"))
        case_to_entries.setdefault(case_id, []).append((label_idx, k, text))

    # sort per case by label index to get stable mask1/mask2 order
    for case_id in case_to_entries:
        case_to_entries[case_id].sort(key=lambda t: t[0])
    return case_to_entries


def resolve_case_paths(case_id: str):
    """
    Returns (ct_path, pet_path, out_dir)
    """
    out_dir = Path(DATA_ROOT) / case_id
    ct_path = out_dir / f"{case_id}{CT_SUFFIX}"
    pet_path = out_dir / f"{case_id}{PET_SUFFIX}"
    if not ct_path.exists():
        print(f"[NOT OK] CT Path missing: {case_id}")
        return None, None, None
    if not pet_path.exists():
        print(f"[NOT OK] PET Path missing: {case_id}")
        return None, None, None
    return str(ct_path), str(pet_path), out_dir


# -------------------------------------------------------------------------
# INFERENCE PER LABEL (save mask1/mask2/...)
# -------------------------------------------------------------------------

def infer_single_mask(model, tokenizer, ct_path: str, pet_path: str, report_text: str, save_path: str):
    global bad_count, good_count
    # Load volumes
    pet_vol, pet_affine, pet_header = load_nii(pet_path)
    ct_vol, _, _ = load_nii(ct_path)

    if pet_vol.shape != ct_vol.shape:
        print(f"PET and CT shapes differ for:\n  PET={pet_path} {pet_vol.shape}\n  CT={ct_path} {ct_vol.shape}")
        bad_count += 1
    else:
        good_count += 1
    return False

    images = prepare_image_tensors(pet_vol, ct_vol)
    ids, mask, token_type_ids = prepare_text_tensors(tokenizer, report_text)

    with torch.no_grad():
        logits = model(images, ids, mask, token_type_ids)  # (1,2,H,W,D)

    probs = torch.softmax(logits, dim=1)
    lesion_prob = probs[:, 1, ...]
    lesion_mask = (lesion_prob > PRED_THRESHOLD).float()

    lesion_np = lesion_mask.squeeze(0).cpu().numpy().astype(np.uint8)

    # Safety crop (kept from your script)
    H, W, D = pet_vol.shape
    lesion_np = lesion_np[:H, :W, :D]

    lesion_nii = nib.Nifti1Image(lesion_np, affine=pet_affine, header=pet_header)
    nib.save(lesion_nii, save_path)
    return True


def run_batch():
    label_json = load_label_json()
    case_to_entries = group_entries_by_case(label_json)

    model, tokenizer = build_model_and_tokenizer()

    total = 0
    correct = 0
    for case_id, entries in case_to_entries.items():
        ct_path, pet_path, out_dir = resolve_case_paths(case_id)
        if ct_path is None or pet_path is None or out_dir is None:
            continue

        # IMPORTANT: You requested:
        # - if multiple labels => save mask1, mask2, ...
        # - if only one label  => still save mask1
        for i, (label_idx, key, text) in enumerate(entries, start=1):
            save_path = str(out_dir / f"mask{i}.nii.gz")
            result = infer_single_mask(
                model=model,
                tokenizer=tokenizer,
                ct_path=ct_path,
                pet_path=pet_path,
                report_text=text,
                save_path=save_path,
            )
            total += 1
            if result:
                correct += 1
                print(f"[OK] {case_id} / {key} -> {save_path}")
            else:
                print(f"[NOT OK] {case_id} / {key}")


    print(f"Done. Wrote {correct} mask(s) out of {total}.")
    print("Shape mismatch", bad_count)
    print("No shape mismatch", good_count)


if __name__ == "__main__":
    run_batch()
