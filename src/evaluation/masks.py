"""Tracking scoring masks, applied the same way to gt and predicted rows.

prepare writes them to outputs/eval_masks/:
- skip_frames.csv (cam,frame): frames with no SCT ground truth (issue D11), dropped.
- ignore_regions.json ({cam: [polygon, ...]}): the area SCT annotates (issue D23).
  Rows whose box bottom-centre (x + w/2, y + h) is outside every polygon are dropped.

    from src.evaluation.masks import apply_masks
    gt = apply_masks(gt, "cam01"); pred = apply_masks(pred, "cam01")
"""
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.data.regions import MASK_DIR, REGION_FILE, SKIP_FILE, inside_region


@lru_cache
def load_masks(mask_dir: Path = MASK_DIR) -> tuple[dict[str, set[int]], dict[str, list]]:
    """({cam: frames to skip}, {cam: polygons})"""
    skip = pd.read_csv(Path(mask_dir) / SKIP_FILE)
    skip_frames = {cam: set(g.frame.astype(int)) for cam, g in skip.groupby("cam")}
    regions = json.loads((Path(mask_dir) / REGION_FILE).read_text(encoding="utf-8"))
    return skip_frames, regions


def apply_masks(df: pd.DataFrame, cam: str, mask_dir: Path = MASK_DIR) -> pd.DataFrame:
    """Rows of a tracks or detections table (frame, x, y, w, h, ...) that tracking scoring keeps."""
    skip_frames, regions = load_masks(Path(mask_dir))
    keep = ~df.frame.isin(skip_frames.get(cam, set())).to_numpy()
    keep &= inside_region(df, regions[cam])
    return df[keep]
