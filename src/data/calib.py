"""Calibration stage: image lists for Hailo int8 quantization, from the train split only.

Writes repo-relative paths, one per line (no images are copied):
  data/interim/calib/det_frames.txt  det_split.calib_frames train detection frames (data/interim/frames/...),
                                     the same number per camera, evenly spaced over the train window
  data/interim/calib/reid_crops.txt  det_split.calib_crops ReID crops with split == train in
                                     outputs/reid_split/gt/crops.csv, the same number per camera and crop-size
                                     quartile, evenly spaced in time within each
The export step applies the deployed preprocessing (resize, letterbox or tiling) to these images;
see docs/pipeline_contract.md ("Calibration sets").
Run from the repo root:  python -m src.data.calib
"""
import json

import numpy as np
import pandas as pd

from src.common.config import ROOT, load_params
from src.data.det_split import COCO_DIR

CALIB_DIR = ROOT / "data" / "interim" / "calib"
FRAME_DIR = "data/interim/frames"


def evenly(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """n rows evenly spaced over df (all rows if it has fewer)."""
    if len(df) <= n:
        return df
    return df.iloc[np.linspace(0, len(df) - 1, n).round().astype(int)]


def calib_frames(images: pd.DataFrame, n: int) -> pd.DataFrame:
    """n // cameras frames per camera, evenly spaced in time."""
    per_cam = n // images.cam.nunique()
    return pd.concat([evenly(g.sort_values("frame"), per_cam) for _, g in images.groupby("cam")])


def calib_crops(crops: pd.DataFrame, n: int) -> pd.DataFrame:
    """Per camera, 4 crop-area quartiles with n // (cameras x 4) crops each, evenly spaced in time."""
    crops = crops.assign(area=crops.w * crops.h)
    crops["size_bin"] = crops.groupby("cam").area.transform(lambda a: pd.qcut(a, 4, labels=False, duplicates="drop"))
    per_bin = n // (crops.cam.nunique() * 4)
    return pd.concat([evenly(g.sort_values(["frame", "path"]), per_bin) for _, g in crops.groupby(["cam", "size_bin"])])


def write_list(path, paths) -> None:
    path.write_text("".join(f"{p}\n" for p in paths), encoding="utf-8", newline="\n")


def main() -> None:
    p = load_params()["det_split"]
    images = pd.DataFrame(json.loads((COCO_DIR / "train.json").read_text(encoding="utf-8"))["images"])
    crops = pd.read_csv(ROOT / "outputs" / "reid_split" / "gt" / "crops.csv")
    crops = crops[crops.split == "train"]

    frames = calib_frames(images, p["calib_frames"])
    picked = calib_crops(crops, p["calib_crops"])
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    write_list(CALIB_DIR / "det_frames.txt", [f"{FRAME_DIR}/{f}" for f in frames.file_name])
    write_list(CALIB_DIR / "reid_crops.txt", picked.path)

    missing = sum(not (ROOT / FRAME_DIR / f).exists() for f in frames.file_name)
    print(f"det_frames.txt: {len(frames)} train frames, {frames.groupby('cam').size().to_dict()}, "
          f"frames {frames.frame.min()}-{frames.frame.max()}, median step {int(frames.frame.diff().median())}"
          + (f"; {missing} not extracted yet (run the frames stage)" if missing else ""))
    print(f"reid_crops.txt: {len(picked)} train crops of {picked.vehicle_id.nunique()} vehicles "
          f"(train has {len(crops)} crops, {crops.vehicle_id.nunique()} vehicles), "
          f"{picked.groupby('cam').size().to_dict()}")
    q = picked.assign(area=picked.w * picked.h)
    print("  crop w x h: " + ", ".join(f"p{int(k * 100)} {int(q.w.quantile(k))}x{int(q.h.quantile(k))}"
                                       for k in (0.05, 0.25, 0.5, 0.75, 0.95))
          + f"; train crops median {int(crops.w.median())}x{int(crops.h.median())}")


if __name__ == "__main__":
    main()
