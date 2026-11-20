"""Prepare stage: turn the RoundaboutHD annotations into the pipeline's `gt` runs.

Writes outputs/detections/gt/<cam>.csv, outputs/tracks/gt/<cam>.csv,
outputs/matching/gt/global_ids.csv and reports/data_quality.json.
Run from the repo root:  python -m src.data.prepare
"""
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import ROOT, load_params
from src.common.io import DET_COLS, GLOBAL_COLS, TRACK_COLS, write_table
from src.data import checks
from src.data.boxes import XYXY, clip_to_xywh, match_by_frame, outside_frame

LABEL_NAME = re.compile(r"(?:^|/)img(\d{6})\.txt$")


def cam_dir(raw: Path, cam: str) -> Path:
    """cam01 -> <raw>/imagesc001"""
    return raw / f"imagesc0{cam[-2:]}"


def read_labels(zip_path: Path) -> tuple[pd.DataFrame, set[str], set[int]]:
    """Read a label zip in memory. imgNNNNNN.txt is frame N + 1.

    Returns (rows `frame, cls, x1, y1, x2, y2, score`, folder names in the zip, frames that have a file).
    """
    rows, folders, frames = [], set(), set()
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            m = LABEL_NAME.search(name)
            if not m:
                continue
            folders.add(name.rsplit("/", 1)[0] if "/" in name else "")
            frame = int(m.group(1)) + 1
            frames.add(frame)
            for line in zf.read(name).decode().splitlines():
                if line.strip():
                    cls, x1, y1, x2, y2, score = line.split()
                    rows.append((frame, int(float(cls)), float(x1), float(y1), float(x2), float(y2), float(score)))
    df = pd.DataFrame(rows, columns=["frame", "cls", *XYXY, "score"])
    return df.sort_values("frame", kind="stable").reset_index(drop=True), folders, frames


def read_sct(path: Path) -> pd.DataFrame:
    """SCT_GT.txt: `frame track_id x1 y1 x2 y2 cls`, in file order."""
    return pd.read_csv(path, sep=r"\s+", header=None, names=["frame", "track_id", *XYXY, "cls"])


def read_multicam(path: Path) -> pd.DataFrame:
    """Multi_CAM_Ground_Turth.txt: `cam id frame x y w h xworld yworld` -> cam (cam01..), mc_id, frame, x1..y2."""
    df = pd.read_csv(path, sep=r"\s+", header=None,
                     names=["cam", "mc_id", "frame", "x", "y", "w", "h", "xworld", "yworld"])
    df["cam"] = df.cam.map(lambda c: f"cam{c:02d}")
    df["x1"], df["y1"], df["x2"], df["y2"] = df.x, df.y, df.x + df.w, df.y + df.h
    return df


def attach_mc_ids(sct: pd.DataFrame, mc: pd.DataFrame, min_iou: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Give each SCT row the Multi_CAM id whose box matches it in the same frame (-1 if none).

    Returns (sct with an `mc_id` column, Multi_CAM rows that matched no SCT box).
    """
    hit, _ = match_by_frame(mc, sct, min_iou)
    sct = sct.copy()
    sct["mc_id"] = -1
    sct.loc[hit[hit >= 0], "mc_id"] = mc.mc_id.to_numpy()[hit >= 0]
    return sct, mc[hit < 0]


def split_id_switches(sct: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Split SCT tracks that Multi_CAM gives two vehicle ids in two separate time spans.

    The later part starts at the first frame of the later Multi_CAM id and gets a new
    track id (max track id on the camera + 1). Tracks whose ids interleave are left alone.
    """
    sct = sct.copy()
    original = sct.track_id.copy()
    splits = []
    next_id = int(sct.track_id.max()) + 1
    for tid, g in sct[sct.mc_id >= 0].groupby("track_id"):
        spans = g.groupby("mc_id").frame.agg(["min", "max"]).sort_values("min")
        if len(spans) < 2 or (spans["min"].to_numpy()[1:] <= spans["max"].to_numpy()[:-1]).any():
            continue
        starts = list(spans["min"].iloc[1:]) + [np.inf]
        for k, mc_id in enumerate(spans.index[1:]):
            part = (original == tid) & (sct.frame >= starts[k]) & (sct.frame < starts[k + 1])
            splits.append(dict(track_id=int(tid), new_track_id=next_id, from_frame=int(starts[k]),
                               mc_before=int(spans.index[k]), mc_after=int(mc_id), rows=int(part.sum())))
            sct.loc[part, "track_id"] = next_id
            next_id += 1
    return sct, splits


def drop_duplicates(sct: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the first row of each (frame, track_id); return (kept, dropped)."""
    dup = sct.duplicated(["frame", "track_id"], keep="first")
    return sct[~dup], sct[dup]


def global_ids(sct: pd.DataFrame, cam: str, single_cam_offset: int) -> tuple[pd.DataFrame, list[dict]]:
    """One global id per track: its Multi_CAM id (majority vote if several),
    or cam_number * single_cam_offset + track_id if it has none."""
    rows, votes = [], []
    for tid, g in sct.groupby("track_id"):
        ids = g.mc_id[g.mc_id >= 0].value_counts()
        if ids.empty:
            gid = int(cam[-2:]) * single_cam_offset + int(tid)
        else:
            gid = int(ids.index[0])
            if len(ids) > 1:
                votes.append(dict(cam=cam, track_id=int(tid), votes={int(k): int(v) for k, v in ids.items()}))
        rows.append((cam, int(tid), gid, ids.empty))
    return pd.DataFrame(rows, columns=[*GLOBAL_COLS, "single_cam"]), votes


def main() -> None:
    params = load_params()
    p = params["prepare"]
    raw = ROOT / params["raw_dir"]
    width, height = p["frame_width"], p["frame_height"]
    multicam = read_multicam(raw / "Multi_CAM_Ground_Turth.txt")
    assert multicam.mc_id.max() < p["single_cam_id_offset"], "single-camera ids would collide with Multi_CAM ids"

    stats, all_globals = {}, []
    for cam in params["cameras"]:
        folder = cam_dir(raw, cam)
        labels, label_folders, label_frames = read_labels(folder / "detection" / "labels_GT.zip")
        labels_test, _, test_frames = read_labels(folder / "detection" / "labels_test.zip")
        sct_raw = read_sct(folder / "SCT" / "SCT_GT.txt")
        mc = multicam[multicam.cam == cam].reset_index(drop=True)

        # detections: every labels_GT box, clipped, score 1.0, class as given
        det = clip_to_xywh(labels, width, height)
        det["score"] = 1.0
        write_table(det, ROOT / "outputs" / "detections" / "gt" / f"{cam}.csv", DET_COLS)

        # tracks: Multi_CAM ids per row -> split id switches -> drop duplicates -> clip
        sct, mc_unmatched = attach_mc_ids(sct_raw, mc, p["match_iou"])
        sct, splits = split_id_switches(sct)
        sct, dropped = drop_duplicates(sct)
        trk = clip_to_xywh(sct, width, height)
        trk["score"], trk["cls"] = 1.0, 2
        trk = trk.sort_values(["frame", "track_id"], kind="stable")
        write_table(trk, ROOT / "outputs" / "tracks" / "gt" / f"{cam}.csv", TRACK_COLS)

        glob, votes = global_ids(sct, cam, p["single_cam_id_offset"])
        all_globals.append(glob)
        with_gid = trk.merge(glob[["track_id", "global_id"]], on="track_id")
        shared = with_gid[with_gid.duplicated(["frame", "global_id"], keep=False)]

        stats[cam] = dict(
            label_files=len(label_frames), label_folders=sorted(label_folders),
            det_rows=len(det), det_outside=int(outside_frame(labels, width, height).sum()),
            sct_rows=len(sct_raw), track_rows=len(trk), sct_outside=int(outside_frame(sct, width, height).sum()),
            tracks_before=int(sct_raw.track_id.nunique()), tracks_after=int(trk.track_id.nunique()),
            splits=splits, dropped=dropped, votes=votes,
            single_cam_tracks=int(glob.single_cam.sum()), global_rows=len(glob), shared_global=shared,
            mc_unmatched=mc_unmatched, mc_rows=len(mc),
            empty_frames=checks.frames_without_rows(sct_raw, p["n_frames"]),
            sct_rows_without_label_file=int((~sct_raw.frame.isin(label_frames)).sum()),
            unmatched_labels=checks.labels_not_in_tracks(labels, sct_raw, p["match_iou"]),
            sct_class=checks.track_classes_from_labels(sct_raw, labels, p["class_iou"]),
            test_not_in_sct=checks.boxes_not_in(labels_test, sct_raw, p["same_box_px"]),
            test_files_missing=p["n_frames"] - len(test_frames),
            world_unknown=int(((mc.xworld == -1) & (mc.yworld == -1)).sum()),
        )
        print(f"{cam}: {len(det)} detections, {len(trk)} track rows, "
              f"{stats[cam]['tracks_after']} tracks, {len(splits)} split, {len(dropped)} duplicates dropped")

    glob = pd.concat(all_globals, ignore_index=True)
    write_table(glob, ROOT / "outputs" / "matching" / "gt" / "global_ids.csv", GLOBAL_COLS)

    issues = checks.quality_log(stats, raw, p)
    out = ROOT / "reports" / "data_quality.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(issues, indent=2), encoding="utf-8")
    checks.print_summary(issues)


if __name__ == "__main__":
    main()
