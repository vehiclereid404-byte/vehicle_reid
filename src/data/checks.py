"""Data checks for the prepare stage and the data quality log (reports/data_quality.json).

prepare calls the count functions and quality_log(). Run alone to print the last log:
    python -m src.data.checks
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import ROOT
from src.data.boxes import XYXY, match_by_frame
from src.data.parked import MAX_CENTRE_MOTION_PX, MIN_FRAMES

LOG_PATH = ROOT / "reports" / "data_quality.json"


# ---------- counts on the raw annotations ----------

def frames_without_rows(df: pd.DataFrame, n_frames: int) -> list[int]:
    """Frames 1..n_frames that have no row in df."""
    return sorted(set(range(1, n_frames + 1)) - set(df.frame))


def labels_not_in_tracks(labels: pd.DataFrame, tracks: pd.DataFrame, min_iou: float) -> int:
    """Detection label boxes with no tracking box (IoU >= min_iou) in the same frame."""
    hit, _ = match_by_frame(labels, tracks, min_iou)
    return int((hit < 0).sum())


def track_classes_from_labels(tracks: pd.DataFrame, labels: pd.DataFrame, min_iou: float) -> dict[int, int]:
    """Class that labels_GT gives each tracking box (best match with IoU >= min_iou): {cls: count}."""
    hit, _ = match_by_frame(tracks, labels, min_iou)
    return {int(k): int(v) for k, v in labels.cls.loc[hit[hit >= 0]].value_counts().items()}


def boxes_not_in(a: pd.DataFrame, b: pd.DataFrame, max_px: float) -> int:
    """Rows of a with no row of b in the same frame whose corners are all within max_px."""
    hit, _ = match_by_frame(a, b, 0.0)
    diff = np.full(len(a), np.inf)
    ok = hit >= 0
    diff[ok] = np.abs(a[XYXY].to_numpy(float)[ok] - b.loc[hit[ok], XYXY].to_numpy(float)).max(1)
    return int((diff >= max_px).sum())


def readme_name_errors(raw: Path) -> list[str]:
    """Names in the dataset README that differ from the files on disk (case-sensitive)."""
    def names(folder: Path) -> set[str]:
        return {p.name for p in folder.iterdir()} if folder.is_dir() else set()

    errors = []
    top, cam1 = names(raw), raw / "imagesc001"
    checks = [
        ("Multi_CAM_Ground_Truth.txt", "Multi_CAM_Ground_Turth.txt", top),
        ("vehicle_statistic.xlsx", "Vehicle_statistic.xlsx", top),
        ("ReID_subset/ (ReID_subset.zip)", "RoundaboutHD_Reid_subset/", top),
        ("SCT/imagesc001_SCT_GT.txt", "SCT/SCT_GT.txt", {f"SCT/{n}" for n in names(cam1 / "SCT")}),
        ("geo-mapping/cam01_undistorted.png", "geo-mapping/cam01.yaml",
         {f"geo-mapping/{n}" for n in names(cam1 / "geo-mapping")}),
    ]
    for readme, actual, present in checks:
        if readme.split(" ")[0].rstrip("/") not in {n.rstrip("/") for n in present} and actual.rstrip("/") in {
                n.rstrip("/") for n in present}:
            errors.append(f"README '{readme}' is '{actual}' on disk")
    reid = raw / "RoundaboutHD_Reid_subset" / "query_images"
    pattern = re.compile(r"^\d{4}_cam0\d_\d{6}\.jpg$")
    if reid.is_dir() and all(pattern.match(n) for n in names(reid)):
        errors.append("README ReID names '{track_id:06d}_{cam_id}_{frame_id}.jpg' are "
                      "'{id:04d}_cam0X_{frame:06d}.jpg' with a 0-based frame")
    return errors


# ---------- the quality log ----------

def _issue(id_: str, title: str, per_camera: dict, fix: str, count: int | None = None) -> dict:
    per_camera = {k: int(v) for k, v in per_camera.items()}
    return {"id": id_, "title": title, "count": int(sum(per_camera.values()) if count is None else count),
            "per_camera": per_camera, "fix_applied": fix}


def _spans(frames: list[int]) -> str:
    """[1, 2, 3, 7] -> '1-3, 7'"""
    out, start = [], None
    for i, f in enumerate(frames):
        if start is None:
            start = f
        if i + 1 == len(frames) or frames[i + 1] != f + 1:
            out.append(f"{start}-{f}" if f != start else f"{f}")
            start = None
    return ", ".join(out)


def quality_log(stats: dict, raw: Path, p: dict) -> list[dict]:
    """Build the list of {id, title, count, per_camera, fix_applied} from prepare's per-camera stats."""
    cams = list(stats)
    s = stats
    splits = {c: s[c]["splits"] for c in cams}
    split_text = {(c, d["track_id"]): d for c in cams for d in splits[c]}

    def split_issue(id_, cam, tid):
        d = split_text.get((cam, tid))
        per = {c: (d["rows"] if c == cam and d else 0) for c in cams}
        fix = (f"split at frame {d['from_frame']}: frames >= {d['from_frame']} became track {d['new_track_id']} "
               f"(Multi_CAM {d['mc_before']} before, {d['mc_after']} after)") if d else "not found; nothing split"
        return _issue(id_, f"ID switch: {cam} track {tid} is two vehicles in Multi_CAM", per, fix)

    other_splits = [d for c in cams for d in splits[c] if (c, d["track_id"]) not in {("cam01", 223), ("cam03", 86)}]
    dropped = {c: s[c]["dropped"] for c in cams}
    dropped_text = "; ".join(
        f"{c} track {t}: {len(g)} rows, frames {_spans(sorted(g.frame))}"
        for c in cams for t, g in dropped[c].groupby("track_id"))
    votes = [v for c in cams for v in s[c]["votes"]]
    readme = readme_name_errors(raw)
    mc_unm = {c: s[c]["mc_unmatched"] for c in cams}
    mc_text = "; ".join(f"{c} Multi_CAM id {i}: frames {_spans(sorted(g.frame))}"
                        for c in cams for i, g in mc_unm[c].groupby("mc_id"))
    empty_text = "; ".join(f"{c}: {_spans(s[c]['empty_frames'])}" for c in cams if s[c]["empty_frames"])
    not_car = {c: sum(v for k, v in s[c]["sct_class"].items() if k != 2) for c in cams}
    folders = {c: ",".join(s[c]["label_folders"]) for c in cams}
    shared = {c: s[c]["shared_global"] for c in cams}
    shared_text = "; ".join(
        f"{c} global id {gid}: tracks {sorted(int(t) for t in g.track_id.unique())}, frames {_spans(sorted(g.frame.unique()))}"
        for c in cams for gid, g in shared[c].groupby("global_id"))

    issues = [
        _issue("D01", "Label files are 0-based: imgNNNNNN.txt is frame N+1",
               {c: s[c]["label_files"] for c in cams}, "frame = N + 1 when reading labels_GT"),
        _issue("D02", "Box formats differ: labels and SCT_GT are x1 y1 x2 y2, Multi_CAM is x y w h",
               {c: s[c]["det_rows"] + s[c]["track_rows"] for c in cams},
               "labels_GT and SCT_GT converted to x, y, w, h; Multi_CAM matched to SCT in x1 y1 x2 y2"),
        _issue("D03", "labels_GT folder inside the zip differs per camera ("
               + "; ".join(f"{c}: {folders[c]}" for c in cams) + ")",
               {c: int(folders[c] != "labels_filtered") for c in cams},
               "prepare reads img*.txt from whichever folder the zip has"),
        _issue("D04", "Parked vehicles are only in the detection labels "
               f"(labels_GT boxes with no SCT box at IoU >= {p['match_iou']})",
               {c: s[c]["unmatched_labels"] for c in cams},
               "kept in gt detections; tracking scoring ignores parked tracks "
               f"(src/data/parked.py: track spans >= {MIN_FRAMES} frames and its box centre moves "
               f"< {MAX_CENTRE_MOTION_PX:g} px)"),
        _issue("D05", "SCT labels every vehicle as class 2 (SCT boxes that labels_GT calls truck, bus or motorcycle)",
               not_car, "gt tracks keep cls 2; detection scoring is class-agnostic"),
        split_issue("D06", "cam01", 223),
        split_issue("D07", "cam03", 86),
        _issue("D08", "SCT tracks with no multi-camera id", {c: s[c]["single_cam_tracks"] for c in cams},
               f"own global id = camera number x {p['single_cam_id_offset']} + track_id"),
        _issue("D09", "Duplicate (frame, track_id) rows in SCT_GT", {c: len(dropped[c]) for c in cams},
               f"kept the first row, dropped the rest: {dropped_text}" if dropped_text else "none found"),
        _issue("D10", f"Boxes outside the {p['frame_width']}x{p['frame_height']} frame",
               {c: s[c]["det_outside"] + s[c]["sct_outside"] for c in cams}, "clipped to the frame"),
        _issue("D11", "Frames with no tracking ground truth", {c: len(s[c]["empty_frames"]) for c in cams},
               f"left empty ({empty_text}); SCT rows whose frame has no labels_GT file: "
               f"{sum(s[c]['sct_rows_without_label_file'] for c in cams)}"),
        _issue("D12", "Multi_CAM rows with no SCT box", {c: len(mc_unm[c]) for c in cams},
               f"skipped: {mc_text}" if mc_text else "none found"),
        _issue("D16", "README names differ from the files", {}, "; ".join(readme) + "; code and contract use the real names",
               count=len(readme)),
        _issue("D17", "labels_test boxes not in SCT_GT (no SCT box within "
               f"{p['same_box_px']} px in the same frame)", {c: s[c]["test_not_in_sct"] for c in cams},
               "left open: labels_test is not used by the pipeline"),
        _issue("D18", "Multi_CAM world coordinates are all -1 (unknown)", {c: s[c]["world_unknown"] for c in cams},
               "left open: world coordinates are not used"),
        _issue("D19", "labels_test.zip is missing label files", {c: s[c]["test_files_missing"] for c in cams},
               "left open: labels_test is not used by the pipeline"),
        _issue("D20", "One global id on two tracks of a camera in the same frame (rows)",
               {c: len(shared[c]) for c in cams},
               f"left open, needs a team decision: {shared_text}" if shared_text else "none found"),
    ]
    if other_splits:
        issues.append(_issue("D21", "Other SCT tracks split at a Multi_CAM id change",
                             {c: len(splits[c]) for c in cams}, "; ".join(str(d) for d in other_splits)))
    if votes:
        issues.append(_issue("D22", "Tracks still mapping to several Multi_CAM ids after the splits",
                             {c: sum(v["cam"] == c for v in votes) for c in cams},
                             "majority vote: " + "; ".join(str(v) for v in votes)))
    return issues


def print_summary(issues: list[dict]) -> None:
    print(f"\nData quality ({len(issues)} issues)")
    for i in issues:
        per = " ".join(f"{k[-2:]}:{v}" for k, v in i["per_camera"].items())
        print(f"  {i['id']} {i['title'][:70]:<70} {i['count']:>8}  [{per}]")


def main() -> None:
    print_summary(json.loads(LOG_PATH.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
