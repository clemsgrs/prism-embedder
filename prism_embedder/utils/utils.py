import json
import torch
import random
import numpy as np
import pandas as pd

from pathlib import Path


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_csv(cfg):
    df = pd.read_csv(cfg.csv)
    if "wsi_path" in df.columns:
        wsi_paths = [Path(x) for x in df.wsi_path.values.tolist()]
    elif "slide_path" in df.columns:
        wsi_paths = [Path(x) for x in df.slide_path.values.tolist()]
    if "mask_path" in df.columns:
        mask_paths = [Path(x) for x in df.mask_path.values.tolist()]
    elif "segmentation_mask_path" in df.columns:
        mask_paths = [Path(x) for x in df.segmentation_mask_path.values.tolist()]
    else:
        mask_paths = [None for _ in wsi_paths]
    return wsi_paths, mask_paths


def update_state_dict(model_dict, state_dict):
    """
    Matches weights between `model_dict` and `state_dict`, accounting for:
    - Key mismatches (missing in model_dict)
    - Shape mismatches (tensor size differences)

    Args:
        model_dict (dict): model state dictionary (expected keys and shapes)
        state_dict (dict): checkpoint state dictionary (loaded keys and values)

    Returns:
        updated_state_dict (dict): Weights mapped correctly to `model_dict`
        msg (str): Log message summarizing the result
    """
    success = 0
    shape_mismatch = 0
    missing_keys = 0
    updated_state_dict = {}
    shape_mismatch_list = []
    missing_keys_list = []
    used_keys = set()
    for model_key, model_val in model_dict.items():
        matched_key = False
        for state_key, state_val in state_dict.items():
            if state_key in used_keys:
                continue
            if model_key == state_key:
                if model_val.size() == state_val.size():
                    updated_state_dict[model_key] = state_val
                    used_keys.add(state_key)
                    success += 1
                    matched_key = True  # key is successfully matched
                    break
                else:
                    shape_mismatch += 1
                    shape_mismatch_list.append(model_key)
                    matched_key = True  # key is matched, but weight cannot be loaded
                    break
        if not matched_key:
            # key not found in state_dict
            updated_state_dict[model_key] = model_val  # Keep original weights
            missing_keys += 1
            missing_keys_list.append(model_key)
    # Log summary
    msg = f"{success}/{len(model_dict)} weight(s) loaded successfully"
    if shape_mismatch_list:
        msg += f"\n{shape_mismatch} weight(s) not loaded due to mismatching shapes: {shape_mismatch_list}"
    if missing_keys_list:
        msg += f"\n{missing_keys} key(s) from checkpoint not found in model: {missing_keys_list}"
    return updated_state_dict, msg


def sanitize_json_content(obj):
    if isinstance(obj, dict):
        return {k: sanitize_json_content(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, np.ndarray)):
        return [sanitize_json_content(v) for v in obj]
    elif isinstance(obj, (str, int, bool, float)):
        return obj
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    else:
        return obj.__repr__()


def write_json_file(*, location, content):
    # Writes a json file with the sanitized content
    content = sanitize_json_content(content)
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))
