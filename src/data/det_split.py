"""Detection split stage: the same contiguous time windows on every camera, with buffer gaps.

The windows are set in params.yaml (`det_split.windows`); frames between windows belong to no split.
Detection frames are the window frames with (frame - 1) % frame_step == 0 (frames 1, 11, 21, ...).

Writes
  data/interim/splits/{train,val,test}.txt  cam,frame: every frame of the window, on all cameras (1-based)
  data/interim/splits/windows.csv           split,first_frame,last_frame (the same on every camera)
  data/interim/coco/{split}.json            COCO labels of the detection frames, 4 classes (COCO ids 2, 3, 5, 7)
  data/interim/coco/{split}_1cls.json       the same boxes as one class (category 2, "vehicle")
  reports/metrics/det_split_gt.json         per split and camera: frames, boxes, moving/parked, classes
Format and use by later stages: docs/pipeline_contract.md ("Detection splits").
Run from the repo root:  python -m src.data.det_split
"""
import json

import numpy as np
import pandas as pd

from src.common.config import ROOT, load_params
from src.common.io import DET_COLS, TRACK_COLS, read_table
from src.data.boxes import match_by_frame
from src.data.checks import link_boxes
from src.data.parked import parked_track_ids
from src.data.regions import MASK_DIR, SKIP_FILE

SPLITS = ["train", "val", "test"]
CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}  # COCO ids kept in the labels
STATES = ["moving_sct", "moving_bg", "parked"]
SPLIT_DIR = ROOT / "data" / "interim" / "splits"
COCO_DIR = ROOT / "data" / "interim" / "coco"
METRICS = ROOT / "reports" / "metrics" / "det_split_gt.json"


def image_id(cam: str, frame: int) -> int:
    """COCO image id: camera number x 100000 + frame (cam02 frame 11 -> 200011)."""
    return int(cam[-2:]) * 100_000 + int(frame)


def check_windows(windows: dict, n_frames: int, min_gap: int) -> None:
    """Windows lie in 1..n_frames and are at least min_gap frames apart."""
    spans = sorted((int(a), int(b), s) for s, (a, b) in windows.items())
    for a, b, s in spans:
        if not 1 <= a <= b <= n_frames:
            raise ValueError(f"window {s} [{a}, {b}] is not inside 1..{n_frames}")
    for (_, b1, s1), (a2, _, s2) in zip(spans[:-1], spans[1:]):
        if a2 - b1 - 1 < min_gap:
            raise ValueError(f"gap between {s1} and {s2} is {a2 - b1 - 1} frames, less than min_gap {min_gap}")


def split_frames(windows: dict, cameras: list[str]) -> pd.DataFrame:
    """split, cam, frame: every frame of every window, on every camera."""
    return pd.DataFrame([(s, cam, f) for s in SPLITS for cam in cameras
                         for f in range(int(windows[s][0]), int(windows[s][1]) + 1)],
                        columns=["split", "cam", "frame"])


def detection_frames(frames: pd.DataFrame, step: int) -> pd.DataFrame:
    """The frames detectors use: (frame - 1) % step == 0."""
    return frames[(frames.frame - 1) % step == 0].reset_index(drop=True)


def read_split(split: str) -> pd.DataFrame:
    """cam, frame of one split (every frame of its window). For scoring later stages on a window:
        test = read_split("test"); rows = df[df.frame.isin(test.frame[test.cam == cam])]
    """
    return pd.read_csv(SPLIT_DIR / f"{split}.txt")


def with_xyxy(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(x1=df.x, y1=df.y, x2=df.x + df.w, y2=df.y + df.h)


def box_states(det: pd.DataFrame, tracks: pd.DataFrame, min_iou: float) -> np.ndarray:
    """State of each gt detection box (one camera, all frames):
    - moving_sct: it has a gt track box (IoU >= min_iou), so moving traffic that SCT annotates,
    - parked: no gt track box, and src/data/parked.py flags it after linking boxes with checks.link_boxes,
    - moving_bg: no gt track box and not parked (background traffic).
    The same rule as checks.region_coverage (issues D04, D23)."""
    det = with_xyxy(det)
    hit, _ = match_by_frame(det, with_xyxy(tracks), min_iou)
    state = np.where(hit >= 0, "moving_sct", "moving_bg").astype(object)
    rest = det[hit < 0].copy()
    rest["track_id"] = link_boxes(rest)
    parked = rest.track_id.isin(parked_track_ids(rest)).to_numpy()
    state[np.flatnonzero(hit < 0)[parked]] = "parked"
    return state


def coco_dict(boxes: pd.DataFrame, images: pd.DataFrame, width: int, height: int, one_class: bool) -> dict:
    """COCO detection dict. images: cam, frame; boxes: cam, frame, x, y, w, h, cls, state.
    file_name is relative to data/interim/frames. Each annotation keeps `state` (moving_sct, moving_bg, parked)."""
    imgs = [dict(id=image_id(c, f), file_name=f"{c}/{f:06d}.jpg", width=width, height=height, cam=c, frame=int(f))
            for c, f in images[["cam", "frame"]].itertuples(index=False)]
    b = boxes.merge(images[["cam", "frame"]], on=["cam", "frame"]).sort_values(["cam", "frame"], kind="stable")
    anns = [dict(id=i + 1, image_id=image_id(c, f), category_id=2 if one_class else int(k),
                 bbox=[float(x), float(y), float(w), float(h)], area=round(float(w * h), 2), iscrowd=0, state=s)
            for i, (c, f, x, y, w, h, k, s) in enumerate(
                b[["cam", "frame", "x", "y", "w", "h", "cls", "state"]].itertuples(index=False))]
    cats = ([dict(id=2, name="vehicle", supercategory="vehicle")] if one_class else
            [dict(id=k, name=n, supercategory="vehicle") for k, n in CLASSES.items()])
    info = dict(description="RoundaboutHD labels_GT boxes (outputs/detections/gt), clipped to the frame; "
                            "image id = camera number x 100000 + frame (1-based)")
    return dict(info=info, images=imgs, annotations=anns, categories=cats)


def split_stats(frames: pd.DataFrame, images: pd.DataFrame, boxes: pd.DataFrame, skip: pd.DataFrame) -> pd.DataFrame:
    """One row per split and camera (and an `all` row per split), counted on the detection frames."""
    b = boxes.merge(images, on=["cam", "frame"])
    cameras = list(images.cam.unique())
    rows = []
    for s in SPLITS:
        for cam in cameras + ["all"]:
            cams = cameras if cam == "all" else [cam]
            im = images[(images.split == s) & images.cam.isin(cams)]
            fr = frames[(frames.split == s) & frames.cam.isin(cams)]
            bb = b[(b.split == s) & b.cam.isin(cams)]
            rows.append(dict(
                split=s, cam=cam, first_frame=int(fr.frame.min()), last_frame=int(fr.frame.max()),
                frames=len(fr), images=len(im), empty_images=len(im) - len(bb.groupby(["cam", "frame"]).size()),
                skip_frames=len(skip.merge(fr, on=["cam", "frame"])),
                boxes=len(bb), boxes_per_image=round(len(bb) / len(im), 2),
                **{st: int((bb.state == st).sum()) for st in STATES},
                **{CLASSES[k]: int((bb.cls == k).sum()) for k in CLASSES},
                box_h_median=float(bb.h.median()), box_h_under_32px_pct=round(100 * float((bb.h < 32).mean()), 1)))
    return pd.DataFrame(rows)


def main() -> None:
    params = load_params()
    p, prep, cameras = params["det_split"], params["prepare"], params["cameras"]
    width, height = prep["frame_width"], prep["frame_height"]
    check_windows(p["windows"], prep["n_frames"], p["min_gap"])

    frames = split_frames(p["windows"], cameras)
    images = detection_frames(frames, p["frame_step"])
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        frames.loc[frames.split == s, ["cam", "frame"]].to_csv(SPLIT_DIR / f"{s}.txt", index=False, lineterminator="\n")
    pd.DataFrame([(s, *p["windows"][s]) for s in SPLITS], columns=["split", "first_frame", "last_frame"]).to_csv(
        SPLIT_DIR / "windows.csv", index=False, lineterminator="\n")

    boxes = []
    for cam in cameras:
        det = read_table(ROOT / "outputs" / "detections" / "gt" / f"{cam}.csv", DET_COLS)
        trk = read_table(ROOT / "outputs" / "tracks" / "gt" / f"{cam}.csv", TRACK_COLS)
        boxes.append(det.assign(cam=cam, state=box_states(det, trk, prep["match_iou"])))
        print(f"{cam}: {len(det)} gt boxes labelled moving_sct / moving_bg / parked")
    boxes = pd.concat(boxes, ignore_index=True)

    COCO_DIR.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        im = images[images.split == s]
        for suffix, one in [("", False), ("_1cls", True)]:
            (COCO_DIR / f"{s}{suffix}.json").write_text(
                json.dumps(coco_dict(boxes, im, width, height, one), separators=(",", ":")), encoding="utf-8")

    skip = pd.read_csv(MASK_DIR / SKIP_FILE)
    stats = split_stats(frames, images, boxes, skip)
    pd.set_option("display.width", 250)
    print(f"\nDetection splits (every {p['frame_step']}th frame of each window; boxes counted on those frames)")
    print(stats.to_string(index=False))
    flat = {f"{r['split']}.{r['cam']}.{k}": v for r in stats.to_dict("records")
            for k, v in r.items() if k not in ("split", "cam")}
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(flat, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
