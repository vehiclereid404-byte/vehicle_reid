"""ReID split stage: an ID-disjoint train/val/test split of the RoundaboutHD ReID crops, along the
detection time windows (det_split.windows in params.yaml), so the whole project uses one split.

Every unique crop of RoundaboutHD_Reid_subset (train, query and test folders together) is linked to
the gt track box it was cut from, gets a vehicle id from outputs/matching/gt/global_ids.csv, and is
put in one split by vehicle (and by group of possibly-same vehicles). Val and test hold the vehicles
seen by 2+ cameras inside their window, with only their crops from that window and one query per
camera; every query has a gallery crop of the same vehicle on another camera. Train holds the
vehicles never seen in the val or test window. Other crops are `excluded`.

Writes outputs/reid_split/gt/crops.csv, outputs/reid_split/gt/checks.json and
reports/metrics/reid_split_gt.json. Format and protocol: docs/pipeline_contract.md ("ReID split").
Run from the repo root:  python -m src.data.reid_split
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.common.config import ROOT, load_params

REID_DIR = "RoundaboutHD_Reid_subset"
FOLDERS = {"train_images": "train", "query_images": "query", "test_images": "gallery"}  # official roles
OUT_DIR = ROOT / "outputs" / "reid_split" / "gt"
METRICS = ROOT / "reports" / "metrics" / "reid_split_gt.json"
COLUMNS = ["path", "cam", "frame", "track_id", "global_id", "vehicle_id", "group", "file_id",
           "official", "split", "role", "w", "h"]

# Identity fixes on top of outputs/matching/gt/global_ids.csv, checked on the crops
# (reports/figures/reid_check_*.jpg). Not in prepare yet: they go to the team first.
SAME_VEHICLE = {161: 78}  # black Chevrolet Colorado: Multi_CAM 78 on cam01, 161 on cam02-03 (ReID file id 592)
TRACK_VEHICLE = {         # Multi_CAM 644 is two vehicles: a red dump truck (cam01-02, cam03 track 78) and
    ("cam03", 526): 87,   # the tan Kia Sorento on cam03 (track 526, the D07 part of track 86) and cam04,
    ("cam04", 81): 87,    # which is Multi_CAM 87 on cam02 (ReID file id 653)
}


# ---------- read the crops ----------

def dhash(im: Image.Image) -> np.ndarray:
    """256-bit difference hash (17 x 16 grey thumbnail) as 4 uint64 words."""
    g = np.asarray(im.convert("L").resize((17, 16), Image.BILINEAR), dtype=np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).ravel()).view(np.uint64)


def scan_crops(reid_dir: Path) -> tuple[pd.DataFrame, np.ndarray, int]:
    """One row per unique image (by md5) over the three folders. File names are
    `{file_id:04d}_cam0X_{frame:06d}.jpg` with a 0-based frame, so frame = N + 1.

    Returns (crops, dhash per crop (N, 4), number of files read).
    """
    rows, hashes, seen = [], [], {}
    for folder, role in FOLDERS.items():
        for p in sorted((reid_dir / folder).glob("*.jpg")):
            data = p.read_bytes()
            md5 = hashlib.md5(data).hexdigest()
            if md5 in seen:  # byte-identical copy: keep the first, note the other official role
                rows[seen[md5]]["official"] += "+" + role
                continue
            file_id, cam, frame0 = p.stem.split("_")
            with Image.open(p) as im:
                w, h = im.size
                hashes.append(dhash(im))
            seen[md5] = len(rows)
            rows.append(dict(path=p.relative_to(ROOT).as_posix(), cam=cam, frame=int(frame0) + 1,
                             file_id=int(file_id), official=role, w=w, h=h))
    n_files = len(rows) + sum(r["official"].count("+") for r in rows)
    return pd.DataFrame(rows), np.stack(hashes), n_files


def read_tracks(cameras: list[str]) -> pd.DataFrame:
    tracks = [pd.read_csv(ROOT / "outputs" / "tracks" / "gt" / f"{cam}.csv").assign(cam=cam) for cam in cameras]
    return pd.concat(tracks, ignore_index=True)


def read_vehicle_stats(path: Path) -> pd.DataFrame:
    """Vehicle_statistic.xlsx -> cam, track_id, make_model (lower case, letters and digits only)."""
    def norm(s: pd.Series) -> pd.Series:
        return s.astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)

    out = []
    for sheet, df in pd.read_excel(path, sheet_name=None).items():
        df = df.dropna(subset=["Number", "Make", "Model"])
        out.append(pd.DataFrame({"cam": f"cam{sheet[-2:]}", "track_id": df.Number.astype(int),
                                 "make_model": norm(df.Make) + " " + norm(df.Model)}))
    return pd.concat(out, ignore_index=True)


# ---------- identity ----------

def link_to_tracks(crops: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    """Add the gt track_id each crop was cut from (NaN if none).

    A crop is img[int(y1):int(y2), int(x1):int(x2)] of an SCT box in its frame (checked on the
    video), so its size equals the box's integer-pixel size. Boxes up to 1 px off are accepted
    when no exact one exists. If several boxes in the frame fit, keep the track that the other
    crops of the same file id on that camera link to.
    """
    t = tracks.assign(bw=(tracks.x + tracks.w).astype(int) - tracks.x.astype(int),
                      bh=(tracks.y + tracks.h).astype(int) - tracks.y.astype(int))
    m = crops.reset_index()[["index", "file_id", "cam", "frame", "w", "h"]].merge(
        t[["cam", "frame", "track_id", "bw", "bh"]], on=["cam", "frame"])
    m["off"] = np.maximum(abs(m.w - m.bw), abs(m.h - m.bh))
    m = m[m.off <= 1]
    m = m[m.off == m.groupby("index").off.transform("min")]
    n = m.groupby("index").track_id.transform("size")
    home = m[n == 1].groupby(["file_id", "cam"]).track_id.agg(set)
    tie = m[n > 1]
    tie = tie[[t in home.get((f, c), ()) for f, c, t in zip(tie.file_id, tie.cam, tie.track_id)]]
    tie = tie[tie.groupby("index").track_id.transform("size") == 1]
    link = pd.concat([m[n == 1], tie]).set_index("index")
    return crops.assign(track_id=link.track_id, size_off=link.off)


def add_vehicle_ids(crops: pd.DataFrame, glob: pd.DataFrame) -> pd.DataFrame:
    """global_id from global_ids.csv; vehicle_id = global_id with SAME_VEHICLE and TRACK_VEHICLE applied."""
    crops = crops.merge(glob, on=["cam", "track_id"], how="left")
    vid = crops.global_id.replace(SAME_VEHICLE)
    for (cam, track), v in TRACK_VEHICLE.items():
        vid[(crops.cam == cam) & (crops.track_id == track)] = v
    return crops.assign(vehicle_id=vid.astype(int))


def file_id_conflicts(crops: pd.DataFrame) -> dict:
    """File ids that cover 2+ vehicles, and vehicles that have 2+ file ids."""
    pairs = crops.groupby(["file_id", "vehicle_id"]).cam.agg(lambda s: sorted(set(s))).reset_index()
    def listing(key, other):
        many = pairs[pairs.groupby(key)[other].transform("size") > 1]
        return {int(k): {int(o): c for o, c in zip(g[other], g.cam)} for k, g in many.groupby(key)}
    return {"file_id_to_vehicles": listing("file_id", "vehicle_id"),
            "vehicle_to_file_ids": listing("vehicle_id", "file_id")}


def near_duplicate_pairs(crops: pd.DataFrame, hashes: np.ndarray, max_bits: int) -> dict:
    """Vehicle pairs with crops on one camera whose dhashes differ in <= max_bits bits.

    The closest pairs are the same vehicle on two trips (Multi_CAM ids are per trip) or the same
    make and model at the same spot. Returns {(a, b): (cam, distance)}, a < b, closest camera.
    """
    pairs = {}
    vid = crops.vehicle_id.to_numpy()
    for cam, idx in crops.groupby("cam").indices.items():
        h = hashes[idx]
        for s in range(0, len(idx), 256):
            d = np.bitwise_count(h[s:s + 256, None, :] ^ h[None, :, :]).sum(-1, dtype=np.int16)
            i, j = np.nonzero((d <= max_bits) & (vid[idx][s:s + 256, None] != vid[idx][None, :]))
            for a, b, dist in zip(vid[idx][s + i], vid[idx][j], d[i, j]):
                key = (int(min(a, b)), int(max(a, b)))
                if key not in pairs or dist < pairs[key][1]:
                    pairs[key] = (cam, int(dist))
    return pairs


def missed_link_pairs(crops: pd.DataFrame, stats: pd.DataFrame, offset: int, window: int) -> dict:
    """(single-camera vehicle, multi-camera vehicle) pairs that may be one vehicle whose Multi_CAM link
    is missing: same make and model in Vehicle_statistic, the single-camera track on a neighbouring
    camera that the multi-camera vehicle is not seen on, and at most `window` frames between the two
    tracks (99% of real neighbour-camera gaps are <= 1,028 frames).
    Returns {(a, b): (cam of the single-camera track, gap in frames)}, a < b.
    """
    t = crops.groupby(["vehicle_id", "cam", "track_id"]).frame.agg(["min", "max"]).reset_index()
    t = t.merge(stats, on=["cam", "track_id"]).assign(cam_no=lambda d: d.cam.str[-2:].astype(int))
    seen = set(zip(t.vehicle_id, t.cam))
    single, multi = t[t.vehicle_id >= offset], t[t.vehicle_id < offset]
    c = single.merge(multi, on="make_model", suffixes=("_s", "_m"))
    gap = np.maximum(c.min_s - c.max_m, c.min_m - c.max_s)
    unseen = pd.Series([(v, cam) not in seen for v, cam in zip(c.vehicle_id_m, c.cam_s)], index=c.index)
    c = c[((c.cam_no_s - c.cam_no_m).abs() == 1) & unseen & gap.between(0, window)].assign(gap=gap)
    return {(int(min(a, b)), int(max(a, b))): (cam, int(g))
            for a, b, cam, g in zip(c.vehicle_id_s, c.vehicle_id_m, c.cam_s, c.gap)}


def split_groups(vehicle_ids, pairs) -> dict[int, int]:
    """Union-find: vehicle id -> group id (smallest vehicle id in its group)."""
    parent = {int(v): int(v) for v in vehicle_ids}
    def root(v):
        while parent[v] != v:
            v = parent[v]
        return v
    for a, b in pairs:
        ra, rb = root(a), root(b)
        parent[max(ra, rb)] = min(ra, rb)
    return {v: root(v) for v in parent}


# ---------- split ----------

def window_of(frames, windows: dict) -> np.ndarray:
    """Window of each 1-based frame: val, train, test (det_split.windows), or gap (between windows)."""
    frames = np.asarray(frames)
    out = np.full(len(frames), "gap", dtype=object)
    for name, (first, last) in windows.items():
        out[(frames >= first) & (frames <= last)] = name
    return out


def qualifying(crops: pd.DataFrame, offset: int) -> set[int]:
    """Vehicles that can be scored on these crops: a Multi_CAM id (< offset) with crops on 2+ cameras and
    2+ crops on at least one of them (that camera gets a query, the other one holds its cross-camera match)."""
    per_cam = crops.groupby(["vehicle_id", "cam"]).size().groupby(level=0).agg(["size", "max"])
    ok = (per_cam.index < offset) & (per_cam["size"] >= 2) & (per_cam["max"] >= 2)
    return {int(v) for v in per_cam.index[ok]}


def assign_splits(crops: pd.DataFrame, tracks: pd.DataFrame, windows: dict, offset: int) -> tuple[pd.Series, pd.DataFrame]:
    """Split along the detection time windows, one split group at a time (tracks: gt rows with `group`).

    - test: a group with a vehicle that qualifies on its test-window crops. The qualifying vehicles keep their
      test-window crops; all other crops of the group (other windows, gaps, other vehicles) are excluded.
    - val: the same with the val window, for the other groups. So a group in both windows is test if it
      qualifies there, else val if it qualifies there, else excluded.
    - excluded: every other group with a gt track row in the val or test window. It can't be scored there,
      and it isn't trained on, because its vehicles are in an evaluation window.
    - train: all other groups, with all their crops (train window and gaps).
    Returns (split per crop: train, val, test or excluded; per-group table).
    """
    win = window_of(crops.frame, windows)
    track_windows = tracks.groupby("group").frame.agg(lambda f: set(window_of(f, windows)))
    scorable = {s: qualifying(crops[win == s], offset) for s in ("test", "val")}
    rows = []
    for g, members in crops.groupby("group").vehicle_id.agg(set).items():
        seen = track_windows[g]
        where = next((s for s in ("test", "val") if members & scorable[s]), None)
        if where is None:
            where = "excluded" if seen & {"val", "test"} else "train"
        rows.append(dict(group=g, vehicles=sorted(members), windows=sorted(seen), split=where,
                         scored=sorted(members & scorable[where]) if where in scorable else []))
    groups = pd.DataFrame(rows).set_index("group")

    group_split = crops.group.map(groups.split).to_numpy()
    split = pd.Series(np.where(group_split == "train", "train", "excluded"), index=crops.index)
    for s in ("val", "test"):
        split[(group_split == s) & (win == s) & crops.vehicle_id.isin(scorable[s]).to_numpy()] = s
    return split, groups


def assign_roles(crops: pd.DataFrame, per_cam: int, rng: np.random.Generator) -> pd.Series:
    """train and excluded crops keep their split as role. In val and test, per vehicle and camera with 2+ crops,
    `per_cam` random crops are queries; all other crops are gallery."""
    role = pd.Series(np.where(crops.split.isin(["train", "excluded"]), crops.split, "gallery"), index=crops.index)
    evals = crops[crops.split.isin(["val", "test"])].sort_values(["vehicle_id", "cam", "frame"])
    for _, g in evals.groupby(["vehicle_id", "cam"]):
        if len(g) >= 2:
            role.loc[rng.choice(g.index.to_numpy(), size=min(per_cam, len(g) - 1), replace=False)] = "query"
    return role


def queries_without_match(crops: pd.DataFrame) -> int:
    """Queries with no gallery crop of the same vehicle on another camera in the same split."""
    q, g = crops[crops.role == "query"], crops[crops.role == "gallery"]
    cams = g.groupby(["split", "vehicle_id"]).cam.agg(set)
    return sum(not (cams.get((s, v), set()) - {c}) for s, v, c in zip(q.split, q.vehicle_id, q.cam))


def crop_size_note(crops: pd.DataFrame) -> dict:
    """Crop sizes against the 256x128 Hailo ReID input (H x W, the person-ReID shape)."""
    aspect = crops.w / crops.h
    return {"crop_w_median": float(crops.w.median()), "crop_h_median": float(crops.h.median()),
            "crop_aspect_wh_median": round(float(aspect.median()), 2),
            "crop_aspect_wh_p10": round(float(aspect.quantile(0.1)), 2),
            "crop_aspect_wh_p90": round(float(aspect.quantile(0.9)), 2),
            "crop_h_below_128_frac": round(float((crops.h < 128).mean()), 3),
            "crop_w_below_128_frac": round(float((crops.w < 128).mean()), 3),
            "crop_w_below_256_frac": round(float((crops.w < 256).mean()), 3)}


def window_checks(crops: pd.DataFrame, tracks: pd.DataFrame, windows: dict, offset: int) -> dict:
    """Per evaluation window: its multi-camera vehicles (gt track rows on 2+ cameras inside it), how many of them
    are in that ReID split, and how many vehicles seen in the window (any gt track row) are ReID train vehicles."""
    split = crops[crops.split != "excluded"].groupby("vehicle_id").split.first()
    tw = tracks[["vehicle_id", "cam"]].assign(window=window_of(tracks.frame, windows))
    out = {}
    for s in ("val", "test"):
        seen = tw[tw.window == s]
        cams = seen[seen.vehicle_id < offset].groupby("vehicle_id").cam.nunique()
        multi = cams.index[cams >= 2]
        out[f"{s}_window_multicam_vehicles"] = len(multi)
        out[f"{s}_window_multicam_in_{s}"] = int((split.reindex(multi) == s).sum())
        out[f"{s}_window_vehicles"] = int(seen.vehicle_id.nunique())
        out[f"train_vehicles_in_{s}_window"] = int((split.reindex(seen.vehicle_id.unique()) == "train").sum())
    return out


def group_table(groups: pd.DataFrame) -> dict:
    return {int(g): dict(vehicles=[int(v) for v in r.vehicles], windows=r.windows, split=r.split,
                         scored=[int(v) for v in r.scored]) for g, r in groups.iterrows()}


def main() -> None:
    params = load_params()
    p, offset = params["reid_split"], params["prepare"]["single_cam_id_offset"]
    windows = params["det_split"]["windows"]
    raw = ROOT / params["raw_dir"]
    rng = np.random.default_rng(params["seed"])
    glob = pd.read_csv(ROOT / "outputs" / "matching" / "gt" / "global_ids.csv")

    crops, hashes, n_files = scan_crops(raw / REID_DIR)
    tracks = read_tracks(params["cameras"])
    crops = link_to_tracks(crops, tracks)
    unlinked = int(crops.track_id.isna().sum())
    keep = crops.track_id.notna().to_numpy()
    crops, hashes = crops[keep].reset_index(drop=True), hashes[keep]
    crops["track_id"] = crops.track_id.astype(int)
    crops = add_vehicle_ids(crops, glob)

    conflicts = file_id_conflicts(crops)
    near = near_duplicate_pairs(crops, hashes, p["near_dup_bits"])
    missed = missed_link_pairs(crops, read_vehicle_stats(raw / "Vehicle_statistic.xlsx"), offset,
                               p["link_window_frames"])
    pairs = list(near) + list(missed)
    group_of = split_groups(crops.vehicle_id.unique(), pairs)
    crops["group"] = crops.vehicle_id.map(group_of)
    tracks = add_vehicle_ids(tracks, glob)
    tracks["group"] = tracks.vehicle_id.map(group_of).fillna(tracks.vehicle_id).astype(int)  # no crops: own group

    crops["split"], groups = assign_splits(crops, tracks, windows, offset)
    crops["role"] = assign_roles(crops, p["queries_per_cam"], rng)
    crops = crops.sort_values(["split", "vehicle_id", "cam", "frame"], kind="stable").reset_index(drop=True)

    # checks: one split per vehicle and per group; crops only from their own window (train: train window or gaps);
    # no vehicle seen in the val or test window is trained on; no possibly-same pair is scored under two labels;
    # every query has a cross-camera match
    kept = crops[crops.split != "excluded"]
    assert kept.groupby("vehicle_id").split.nunique().max() == 1
    assert kept.groupby("group").split.nunique().max() == 1
    win = window_of(kept.frame, windows)
    assert np.where(kept.split == "train", np.isin(win, ["train", "gap"]), win == kept.split).all()
    by_window = window_checks(crops, tracks, windows, offset)
    assert by_window["train_vehicles_in_val_window"] == 0 and by_window["train_vehicles_in_test_window"] == 0
    vsplit = kept.groupby("vehicle_id").split.first()
    both_scored = [(a, b) for a, b in pairs if vsplit.get(a) in ("val", "test") and vsplit.get(b) in ("val", "test")]
    assert not both_scored, f"possibly-same vehicles scored under two labels: {both_scored}"
    cross_pairs = sum(a in vsplit and b in vsplit and vsplit[a] != vsplit[b] for a, b in pairs)
    no_match = queries_without_match(crops)
    assert no_match == 0, f"{no_match} queries without a cross-camera gallery crop"

    scored = crops.vehicle_id.isin(vsplit.index[vsplit.isin(["val", "test"])])
    reason = pd.Series(np.select(
        [crops.split != "excluded", scored, crops.group.map(groups.split).isin(["val", "test"])],
        ["", "scored_vehicle_other_window", "other_vehicle_of_scored_group"], "group_in_eval_window"))
    straddle = groups[groups.windows.map(lambda w: {"val", "test"} <= set(w))]
    multi = groups[groups.vehicles.map(len) > 1]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    crops[COLUMNS].to_csv(OUT_DIR / "crops.csv", index=False, lineterminator="\n")

    metrics = {"files": n_files, "unique_crops": len(crops) + unlinked,
               "byte_identical_dropped": n_files - len(crops) - unlinked, "unlinked_dropped": unlinked,
               "crops_size_off_1px": int((crops.size_off == 1).sum()),
               "vehicles": int(crops.vehicle_id.nunique()),
               "multi_cam_vehicles": int(crops.loc[crops.vehicle_id < offset, "vehicle_id"].nunique()),
               "file_ids_with_2plus_vehicles": len(conflicts["file_id_to_vehicles"]),
               "vehicles_with_2plus_file_ids": len(conflicts["vehicle_to_file_ids"]),
               "near_dup_pairs": len(near), "missed_link_pairs": len(missed),
               "possibly_same_pairs_across_splits": int(cross_pairs), "queries_without_cross_cam_match": no_match,
               "groups_in_val_and_test_window": len(straddle), **by_window}
    for s in ["train", "val", "test"]:
        part = crops[crops.split == s]
        metrics[f"{s}_ids"] = int(part.vehicle_id.nunique())
        metrics[f"{s}_crops"] = len(part)
        if s != "train":
            metrics[f"{s}_query"] = int((part.role == "query").sum())
            metrics[f"{s}_gallery"] = int((part.role == "gallery").sum())
    metrics["excluded_crops"] = int((crops.split == "excluded").sum())
    metrics["excluded_only_ids"] = int(crops.vehicle_id.nunique() - kept.vehicle_id.nunique())
    metrics |= {f"excluded_crops.{k}": int(v) for k, v in reason[reason != ""].value_counts().sort_index().items()}
    metrics |= crop_size_note(crops)
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8", newline="\n")

    checks = {**conflicts,
              "near_duplicate_pairs": [dict(a=a, b=b, cam=c, bits=d) for (a, b), (c, d) in sorted(near.items())],
              "missed_link_pairs": [dict(single=a if a >= offset else b, multi=b if a >= offset else a, cam=c, gap=g)
                                    for (a, b), (c, g) in sorted(missed.items())],
              "groups": group_table(multi), "groups_in_val_and_test_window": group_table(straddle),
              "identity_fixes": {"same_vehicle": {str(k): v for k, v in SAME_VEHICLE.items()},
                                 "track_vehicle": {f"{c}/{t}": v for (c, t), v in TRACK_VEHICLE.items()}}}
    (OUT_DIR / "checks.json").write_text(json.dumps(checks, indent=1) + "\n", encoding="utf-8", newline="\n")

    print(f"{n_files} files -> {metrics['unique_crops']} unique crops, {unlinked} unlinked, "
          f"{metrics['vehicles']} vehicles ({metrics['multi_cam_vehicles']} multi-camera)")
    print(f"file ids covering 2+ vehicles: {conflicts['file_id_to_vehicles']}")
    print(f"vehicles with 2+ file ids: {conflicts['vehicle_to_file_ids']}")
    print(f"{len(near)} near-duplicate and {len(missed)} missed-link vehicle pairs -> {len(multi)} groups "
          f"({int(multi.vehicles.map(len).sum())} vehicles); their splits: {multi.split.value_counts().to_dict()}")
    print(f"groups in both the val and the test window: {group_table(straddle)}")
    print(pd.crosstab(crops.split, crops.role, margins=True).to_string())
    print(crops.groupby("split").vehicle_id.nunique().to_string())
    print("excluded crops: " + ", ".join(f"{k[15:]} {v}" for k, v in metrics.items() if k.startswith("excluded_crops.")))
    print("windows: " + ", ".join(f"{k} {v}" for k, v in by_window.items()))


if __name__ == "__main__":
    main()
