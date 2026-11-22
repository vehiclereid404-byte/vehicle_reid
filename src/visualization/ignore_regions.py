"""Check images for the tracking ignore regions (issue D23).

Draws each camera's polygons on one video frame, with the bottom-centres of the gt track boxes
(green) and of the labels_GT boxes that have no gt track box (red), every 10th frame.
Writes reports/figures/ignore_region_<cam>.jpg, about 1280 px wide. Needs the videos.
    python -m src.visualization.ignore_regions [--frame 3000]
"""
import argparse

import cv2
import numpy as np
import pandas as pd

from src.common.config import ROOT, load_params
from src.common.io import DET_COLS, TRACK_COLS, read_table
from src.common.video import iter_frames, video_path
from src.data.boxes import match_by_frame
from src.evaluation.masks import load_masks
from src.data.regions import bottom_centre

GREEN, RED, YELLOW = (0, 200, 0), (0, 0, 255), (0, 255, 255)


def with_xyxy(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(x1=df.x, y1=df.y, x2=df.x + df.w, y2=df.y + df.h)


def draw(cam: str, frame_no: int, width: int = 1280, every: int = 10) -> np.ndarray:
    params = load_params()
    _, image = next(iter_frames(video_path(ROOT / params["raw_dir"], cam), start=frame_no, stop=frame_no))
    tracks = with_xyxy(read_table(ROOT / "outputs" / "tracks" / "gt" / f"{cam}.csv", TRACK_COLS))
    dets = with_xyxy(read_table(ROOT / "outputs" / "detections" / "gt" / f"{cam}.csv", DET_COLS))
    hit, _ = match_by_frame(dets, tracks, params["prepare"]["match_iou"])
    for df, colour in [(dets[hit < 0], RED), (tracks, GREEN)]:
        for x, y in bottom_centre(df[df.frame % every == 0]).astype(int):
            cv2.circle(image, (int(x), int(y)), 4, colour, -1)
    _, regions = load_masks()
    polygons = [np.asarray(p, dtype=np.int32) for p in regions[cam]]
    cv2.polylines(image, polygons, True, YELLOW, 8)
    scale = width / image.shape[1]
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    legend = (f"{cam} frame {frame_no}: region (yellow), bottom-centres every {every}th frame of "
              "gt track boxes (green) and labels_GT boxes with no track box (red)")
    cv2.rectangle(image, (0, 0), (image.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(image, legend, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=int, default=3000)
    args = parser.parse_args()
    out_dir = ROOT / "reports" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for cam in load_params()["cameras"]:
        path = out_dir / f"ignore_region_{cam}.jpg"
        cv2.imwrite(str(path), draw(cam, args.frame), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(path)


if __name__ == "__main__":
    main()
