"""Frames stage: extract the detection frames from the videos and write YOLO labels next to them.

The images to extract and their boxes come from data/interim/coco/{split}.json (det_split stage).
Each video is read once, sequentially from frame 1 (src/common/video.py:iter_frames, no seeking per frame).

Writes (not cached or pushed by DVC: several GB, rebuilt from the videos)
  data/interim/frames/<cam>/<frame:06d>.jpg  1-based frame, full 3840x2160, JPEG quality det_split.jpeg_quality
  data/interim/frames/<cam>/<frame:06d>.txt  YOLO labels `class cx cy w h` (normalised); class 0 car, 1 motorcycle,
                                             2 bus, 3 truck (COCO 2, 3, 5, 7). One class: train with single_cls=True.
  data/interim/frames/yolo_{split}.txt       image list per split (./<cam>/<frame>.jpg)
  data/interim/frames/yolo.yaml              YOLO dataset file (ultralytics, YOLOv5-v9 repos)
  data/interim/frames/index.csv              cam,frame,split,path,bytes,luma (mean grey level 0-255)
  reports/metrics/frames_gt.json             images, MB and seconds per camera
Run from the repo root:  python -m src.data.frames [--cams cam01 ...]
"""
import argparse
import json
import time

import cv2
import pandas as pd

from src.common.config import ROOT, load_params
from src.common.video import iter_frames, video_path
from src.data.det_split import CLASSES, COCO_DIR, SPLITS

FRAME_DIR = ROOT / "data" / "interim" / "frames"
METRICS = ROOT / "reports" / "metrics" / "frames_gt.json"
YOLO_CLASS = {k: i for i, k in enumerate(CLASSES)}  # COCO id -> YOLO class (2->0, 3->1, 5->2, 7->3)


def read_coco(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(images: id, file_name, cam, frame, width, height; annotations: image_id, category_id, bbox)."""
    coco = json.loads((COCO_DIR / f"{split}.json").read_text(encoding="utf-8"))
    return pd.DataFrame(coco["images"]).assign(split=split), pd.DataFrame(coco["annotations"])


def yolo_lines(anns: pd.DataFrame, width: int, height: int) -> list[str]:
    """COCO annotations of one image -> `class cx cy w h`, normalised to 0..1."""
    lines = []
    for k, (x, y, w, h) in zip(anns.category_id, anns.bbox):
        lines.append(f"{YOLO_CLASS[k]} {(x + w / 2) / width:.6f} {(y + h / 2) / height:.6f} "
                     f"{w / width:.6f} {h / height:.6f}")
    return lines


def extract_camera(cam: str, images: pd.DataFrame, raw_dir, step: int, quality: int) -> list[dict]:
    """Decode the camera's video once and save the wanted frames. Returns one index row per image."""
    wanted = images.set_index("frame")
    out_dir = FRAME_DIR / cam
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame, image in iter_frames(video_path(raw_dir, cam), start=1, stop=int(wanted.index.max()), step=step):
        if frame not in wanted.index:
            continue
        path = out_dir / f"{frame:06d}.jpg"
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError(f"JPEG encoding failed: {cam} frame {frame}")
        path.write_bytes(buf.tobytes())
        luma = float(cv2.cvtColor(cv2.resize(image, (960, 540), interpolation=cv2.INTER_AREA),
                                  cv2.COLOR_BGR2GRAY).mean())
        rows.append(dict(cam=cam, frame=frame, split=wanted.at[frame, "split"],
                         path=path.relative_to(ROOT).as_posix(), bytes=len(buf), luma=round(luma, 2)))
    missing = set(wanted.index) - {r["frame"] for r in rows}
    if missing:
        raise RuntimeError(f"{cam}: {len(missing)} frames not read from the video, first {min(missing)}")
    return rows


def write_lines(path, lines) -> None:
    """Text file with LF line endings on every OS."""
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8", newline="\n")


def write_yolo(images: pd.DataFrame, anns: pd.DataFrame) -> None:
    """Label file next to each image (empty if it has no box), one image list per split, and yolo.yaml."""
    by_image = dict(tuple(anns.groupby("image_id")))
    for im in images.itertuples(index=False):
        g = by_image.get(im.id, anns.iloc[:0])
        write_lines((FRAME_DIR / im.file_name).with_suffix(".txt"), yolo_lines(g, im.width, im.height))
    for s in SPLITS:
        write_lines(FRAME_DIR / f"yolo_{s}.txt", [f"./{f}" for f in images.file_name[images.split == s]])
    write_lines(FRAME_DIR / "yolo.yaml", [
        "# Written by python -m src.data.frames. Labels sit next to the images (<cam>/<frame>.txt).",
        "# One class (vehicle): train with single_cls=True. Scoring is class-agnostic (contract, D05).",
        f"path: {FRAME_DIR.as_posix()}",
        *[f"{s}: yolo_{s}.txt" for s in SPLITS],
        "names:", *[f"  {i}: {CLASSES[k]}" for k, i in YOLO_CLASS.items()],
        "coco_ids:  # YOLO class -> COCO category id (contract `cls`)",
        *[f"  {i}: {k}" for k, i in YOLO_CLASS.items()]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cams", nargs="*", help="only these cameras (default: all)")
    args = parser.parse_args()
    params = load_params()
    p, raw_dir = params["det_split"], ROOT / params["raw_dir"]
    cameras = args.cams or params["cameras"]
    parts = [read_coco(s) for s in SPLITS]
    images = pd.concat([im for im, _ in parts], ignore_index=True)
    anns = pd.concat([a for _, a in parts], ignore_index=True)
    if ((images.frame - 1) % p["frame_step"]).any():
        raise ValueError("COCO images are not on the frame_step grid; rerun the det_split stage")

    index, metrics = [], {}
    for cam in cameras:
        start = time.perf_counter()
        rows = extract_camera(cam, images[images.cam == cam], raw_dir, p["frame_step"], p["jpeg_quality"])
        seconds = time.perf_counter() - start
        mb = sum(r["bytes"] for r in rows) / 1e6
        index += rows
        metrics.update({f"{cam}.images": len(rows), f"{cam}.mb": round(mb, 1), f"{cam}.seconds": round(seconds, 1)})
        print(f"{cam}: {len(rows)} frames, {mb:,.0f} MB, {seconds:.0f} s")

    write_yolo(images[images.cam.isin(cameras)], anns)
    index = pd.DataFrame(index)
    index.to_csv(FRAME_DIR / "index.csv", index=False, lineterminator="\n")
    metrics.update({"all.images": len(index), "all.mb": round(index.bytes.sum() / 1e6, 1),
                    "all.seconds": round(sum(v for k, v in metrics.items() if k.endswith(".seconds")), 1)})
    print(index.groupby(["split", "cam"]).agg(images=("frame", "size"), mb=("bytes", lambda b: round(b.sum() / 1e6)),
                                              luma=("luma", "mean")).round(1).to_string())
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(metrics, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
