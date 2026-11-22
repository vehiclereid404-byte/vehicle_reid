"""Loaders for RoundaboutHD raw annotations (shared by notebooks and make_dataset.py)."""
import hashlib
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "RoundaboutHD"
CAMS = [1, 2, 3, 4]
REID_SPLITS = {"train": "train_images", "query": "query_images", "test": "test_images"}
REID_NAME = re.compile(r"^(\d+)_cam(\d+)_(\d+)\.jpg$")


def load_sct(cams=CAMS):
    """Single-camera tracking GT. Columns a,b,c,d kept raw: box format is checked in notebook 1."""
    cols = ["frame", "track_id", "a", "b", "c", "d", "cls"]
    dfs = [pd.read_csv(RAW / f"imagesc{c:03d}/SCT/SCT_GT.txt", sep=r"\s+", header=None, names=cols).assign(cam=c)
           for c in cams]
    return pd.concat(dfs, ignore_index=True)


def load_mcam():
    """Multi-camera GT: cam id frame x1 y1 w h xworld yworld (-1 = no world coords)."""
    cols = ["cam", "id", "frame", "x1", "y1", "w", "h", "xworld", "yworld"]
    return pd.read_csv(RAW / "Multi_CAM_Ground_Turth.txt", sep=r"\s+", header=None, names=cols)


def load_det_zip(cam, kind="GT"):
    """Per-frame detection labels from labels_{kind}.zip, read without extracting.
    Frame index is taken from the file name imgNNNNNN.txt."""
    rows = []
    with zipfile.ZipFile(RAW / f"imagesc{cam:03d}/detection/labels_{kind}.zip") as z:
        for name in z.namelist():
            if not name.endswith(".txt"):
                continue
            frame = int(Path(name).stem[3:])
            for line in z.read(name).decode().split("\n"):
                if line.strip():
                    rows.append([frame, *map(float, line.split())])
    ncol = max(len(r) for r in rows)
    return pd.DataFrame(rows, columns=["frame"] + [f"v{i}" for i in range(ncol - 1)]).assign(cam=cam)


def parse_reid_name(name):
    """'0003_cam01_000311.jpg' -> (id, cam, frame); None if it doesn't match."""
    m = REID_NAME.match(name)
    return tuple(map(int, m.groups())) if m else None


def load_reid_index():
    """One row per ReID crop: path, id, cam, frame, split (unparseable names get NaN)."""
    rows = []
    for split, folder in REID_SPLITS.items():
        for p in sorted((RAW / "RoundaboutHD_Reid_subset" / folder).iterdir()):
            if p.name.startswith("."):
                continue
            parsed = parse_reid_name(p.name) or (None, None, None)
            rows.append([p.relative_to(ROOT).as_posix(), *parsed, split])  # "/" on every OS
    return pd.DataFrame(rows, columns=["path", "id", "cam", "frame", "split"])


def add_image_info(r):
    """Add crop width/height (header only) and md5 of the file bytes."""
    r = r.copy()
    r[["w", "h"]] = [Image.open(ROOT / p).size for p in r.path]
    r["md5"] = [hashlib.md5((ROOT / p).read_bytes()).hexdigest() for p in r.path]
    return r


def link_reid(r, tol=3):
    """Link crops to SCT track_id and Multi-CAM global id (gid).
    ReID frame f (0-based) <-> SCT frame f+1; match by (cam, frame) and crop size within `tol` px.
    SCT track -> gid via identical boxes in the Multi-CAM GT (majority vote per track)."""
    sct = load_sct().rename(columns={"a": "x1", "b": "y1", "c": "x2", "d": "y2"})
    sct["bw"], sct["bh"] = sct.x2 - sct.x1, sct.y2 - sct.y1
    j = r[["path", "cam", "frame", "w", "h"]].assign(frame=r.frame + 1).merge(
        sct[["cam", "frame", "track_id", "bw", "bh"]], on=["cam", "frame"])
    j["err"] = (j.w - j.bw).abs() + (j.h - j.bh).abs()
    best = j.sort_values("err").drop_duplicates("path")
    r = r.merge(best.loc[best.err <= tol, ["path", "track_id"]], on="path", how="left")

    mcam = load_mcam()
    mb = mcam.assign(x2=mcam.x1 + mcam.w, y2=mcam.y1 + mcam.h).round(1)
    g = sct.round(1).merge(mb, on=["cam", "frame", "x1", "y1", "x2", "y2"])
    track2gid = g.groupby(["cam", "track_id"]).id.agg(lambda s: s.mode()[0])
    r["gid"] = [track2gid.get((c, t), np.nan) if pd.notna(t) else np.nan for c, t in zip(r.cam, r.track_id)]
    return r
