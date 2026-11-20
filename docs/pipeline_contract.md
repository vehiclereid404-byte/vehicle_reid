# Pipeline contract

Every stage agent reads this before writing code. It fixes how stages talk to
each other, so any model can be swapped without touching the other stages.
If your stage needs a change here, stop and report the proposed change; don't
edit other stages to make it fit.

All deployed models must run on a **Raspberry Pi 5 + Hailo-8L**. See
`docs/deployment_target.md` for what that rules out and what every model must report.

## Data flow

```
data/raw/RoundaboutHD ─ prepare ─► data/interim/          (cleaned GT, frame subsets, splits)
                                   outputs/*/gt/          (ground truth in pipeline formats)
videos ─ detect ─► outputs/detections/<run>/<cam>.csv
       ─ track  ─► outputs/tracks/<run>/<cam>.csv
       ─ reid   ─► outputs/reid/<run>/<cam>.npz
       ─ match  ─► outputs/matching/<run>/global_ids.csv
       ─ od     ─► outputs/od/<run>/<cam>.csv, corridor.csv
web app reads any of the above
```

Each stage can read the `gt` run instead of the previous stage's output.
The prepare stage writes `outputs/detections/gt`, `outputs/tracks/gt` and
`outputs/matching/gt` from the annotations, so nobody waits on anyone. Later
you switch `input:` in `params.yaml` to a real run.

## Conventions

- **Cameras:** `cam01`..`cam04`, stored in `imagesc001`..`imagesc004`.
- **Frames:** 1-based integers matching `SCT/SCT_GT.txt` (1..9000).
  `src/common/video.py:iter_frames` yields the same numbers (checked on all 4 videos).
  Label files and ReID crop names are 0-based: `img000000.txt` and `*_000000.jpg` are frame 1 (frame = N + 1).
- **Boxes:** absolute pixels in the full 3840×2160 frame (all 4 videos).
  `x, y` = top-left corner, `w, h` = size.
  Raw labels and `SCT_GT.txt` are `x1 y1 x2 y2`. `Multi_CAM_Ground_Turth.txt` is already `x y w h`.
  Only the prepare stage converts them. Some SCT boxes run up to 11 px past the frame edge; prepare clips them.
- **Class:** COCO ids (`2` = car). All SCT ground truth is class 2, but there it means "vehicle":
  in `labels_GT`, about 18% of SCT boxes are trucks (`7`) and a few are buses (`5`).
  So **detection scoring is class-agnostic**: every box counts as "vehicle", and `cls` is kept for information only.
- **Run:** a folder name under `outputs/<stage>/`. `gt` is reserved for ground truth.

## Files between stages

| File | Content | Written by |
|---|---|---|
| `outputs/detections/<run>/<cam>.csv` | `frame,x,y,w,h,score,cls` | detection |
| `outputs/tracks/<run>/<cam>.csv` | `frame,track_id,x,y,w,h,score,cls` | tracking |
| `outputs/reid/<run>/<cam>.npz` | `track_id` (N,), `emb` (N, D) float32, L2-normalised, one per track | reid |
| `outputs/matching/<run>/global_ids.csv` | `cam,track_id,global_id` | matching |
| `outputs/od/<run>/<cam>.csv`, `corridor.csv` | count matrices, rows = origin, cols = destination | od |
| `reports/metrics/<stage>_<run>.json` | flat `{"metric": number}` | each stage, via `src/evaluation` |

Read and write these with `src/common/io.py`, so the column checks stay in one place.

## Ground truth runs (`gt`, written by `python -m src.data.prepare`)

- `outputs/detections/gt/<cam>.csv`: every `labels_GT` box, including parked vehicles.
  Clipped to the frame, `score` = 1.0, `cls` as labelled (2, 3, 5, 7).
- `outputs/tracks/gt/<cam>.csv`: `SCT_GT` clipped to the frame, `score` = 1.0, `cls` = 2.
  For a duplicate `(frame, track_id)`, the first row is kept.
- **ID switches:** Multi_CAM sometimes gives one SCT track two vehicle ids in separate time spans.
  Such a track is split at the first frame of the later id, and the later part gets track id = max track id on that camera + 1.
  Here: cam01 223 → 518 from frame 4075, and cam03 86 → 526 from frame 1288.
- `outputs/matching/gt/global_ids.csv`: one row per gt track.
  - `global_id` = the Multi_CAM id of its boxes (IoU ≥ 0.5; majority vote if there are several).
  - A track with no Multi_CAM id gets `camera number × 10000 + track_id` (cam02 track 7 → 20007).
    So `global_id ≥ 10000` means a vehicle seen by one camera only.
- **Parked vehicles:** SCT has none, but `labels_GT` and every detector do.
  Tracking scoring first drops the predicted tracks that `src/data/parked.py:parked_track_ids` flags, the same way for every tracker.
  - A track is parked if it spans ≥ 150 frames (10 s) and its box centre moves < 30 px.
    Motion is measured as the diagonal of the 5th–95th percentile range of the centre.
  - On the data it flags 0 of 1,044 SCT tracks. Its 10-s minimum keeps short pieces of slow traffic safe.
  - It covers 82% of the `labels_GT` boxes that have no SCT box. The rest are mostly distant traffic
    outside the roundabout, which SCT doesn't annotate, so trackers are still charged for those.
- `reports/data_quality.json`: a list of `{id, title, count, per_camera, fix_applied}` with stable ids
  (`D01`, `D02`, ...). `python -m src.data.checks` prints it.

## Swappable components

| Stage | Interface | Method |
|---|---|---|
| detection | `src/detection/base.py:Detector` | `detect(frames) -> list of (N, 6)` |
| tracking | `src/tracking/base.py:Tracker` | `reset()`, `update(dets, frame) -> (M, 7)` |
| reid | `src/reid/base.py:ReIDModel` | `embed(crops) -> (N, D)` |
| matching | `src/matching/base.py:Matcher` | `match(tracks, embeddings) -> DataFrame` |

- Put one implementation in each file: `src/<stage>/<name>.py`.
- Choose it in `params.yaml` and build it with `src.common.config.build(params[stage]["model"])`:
  ```yaml
  detection:
    model:
      class: src.detection.yolov9:YOLOv9Detector
      args: {weights: models/detection/yolov9/best.pt}
  ```
- The stage entry point `src/<stage>/run.py` loads params, builds the model, reads
  inputs, writes outputs and calls the scorer. It never imports a specific implementation.
- Training code sits next to the model (`src/<stage>/<name>_train.py`).
  Weights go in `models/<stage>/<name>/`.

## Rules for all stages

- **Keep code simple.** Use plain functions and one class per model, with no frameworks of
  our own and no deep class hierarchies. Someone should be able to read a file in five minutes.
- **Use original repos.** Install third-party models from their original repo or package.
  If you must vendor one, put it in `third_party/<repo>/` at a pinned commit.
  Add a row (name, source, licence, commercial use?) to `docs/licenses.md`.
- **Keep settings in `params.yaml`.** Each stage edits only its own section.
  Write paths relative to the repo root.
- **One DVC stage per pipeline stage in `dvc.yaml`.** List code and input files under `deps`,
  your params section under `params`, your output folder under `outs`, and your metrics
  JSON under `metrics` with `cache: false`.
- **Git holds code only.** Data, weights and outputs go through DVC to DagsHub,
  which has a 20 GB cap. Push only cleaned annotations, splits and final weights.
  The raw 4K videos stay on the shared drive.
- **Log experiments to MLflow locally**, in `mlruns/` (gitignored).
- **Score with `src/evaluation/`** (owned by the data-eval agent), so every model is
  measured the same way.
- **Stay in your own folders.** Don't edit another stage's folder; report what you need instead.
- **Test on a small slice first,** for example 300 frames of one camera, before full runs.

## Ownership

| Agent | Owns |
|---|---|
| `data-eval` | `src/data/`, `src/evaluation/`, `src/visualization/`, `docs/licenses.md`, DVC stage `prepare` |
| `detection` | `src/detection/`, `models/detection/`, DVC stage `detect` |
| `tracking` | `src/tracking/`, DVC stage `track` |
| `reid` | `src/reid/`, `models/reid/`, DVC stage `reid` |
| `matching` | `src/matching/`, DVC stage `match` |
| `od-matrix` | `src/od/`, `configs/zones/`, DVC stage `od` |
| `webapp` | `src/webapp/` |

Shared files (`src/common/`, this file) take additive changes only, and
each change must be reported. The old `process_data`, `train` and `eval` stages
(cookiecutter leftovers) have been replaced by `prepare`, and the empty template scripts are deleted.

## Dataset facts (RoundaboutHD)

- 4 cameras, one `video.mp4` (1.2–2.3 GB) per camera: 3840×2160, 15 fps, 9000 frames (10 min).
  The README calls them synchronised. Cross-camera travel times agree with that.
- The cameras form a corridor cam01 – cam02 – cam03 – cam04. Median travel time between neighbouring cameras is 30–50 s
  (1↔2 ≈ 450–500 frames, 2↔3 ≈ 720, 3↔4 ≈ 600). The 10th percentile is ≥ 26 s; the shortest is 8 s (2→1).
- Each camera folder has `detection/labels_GT.zip`, `labels_xy.zip` and `labels_test.zip`.
  Each zip holds `img000000.txt`..`img008999.txt`, with lines `cls x1 y1 x2 y2 last` and no empty files.
  Inside, the folders are `labels_filtered/` (cam01–02) or `labels_corrected/` (cam03–04), then `labels_xy/` and `labels_sct2det/`.
  - `labels_xy`: raw YOLOv12x output in integer pixels. `last` = confidence (0.30–0.98).
  - `labels_GT`: `labels_xy` with false positives removed and 1,430 boxes added by hand (`last` = 1.0).
    It includes parked vehicles, so it has 2.5× as many boxes as SCT.
    Classes: 2: 499,116, 7: 75,258, 5: 226, 3: 7.
  - `labels_test`: the SCT_GT boxes per frame, class 2 only, `last` always 1.
    cam04 lacks `img004099`..`img004228`. On cam01, 3% of its boxes are not in SCT_GT.
- `SCT/SCT_GT.txt` is the single-camera tracking ground truth: `frame track_id x1 y1 x2 y2 cls`, frames 1..9000.
  - Track ids are per camera and have gaps (cam01: 262 tracks, ids 1..517).
  - Duplicate `(frame, track_id)` rows with different boxes: cam01 2 (track 214), cam03 10 (431), cam04 14 (105, 314).
  - Frames with no rows: cam02 2946, cam04 4099–4229.
- `Multi_CAM_Ground_Turth.txt` (the misspelling is in the file name):
  `cam id frame x y w h xworld yworld`.
  - Boxes and frames are SCT_GT's: 99.97% of rows equal an SCT box in the same frame.
    `id` is a cross-camera vehicle id, unrelated to SCT track ids, so map it per camera by box match.
  - 311 ids, 305 of them seen by 2–4 cameras. Per camera, 179/262, 221/244, 245/279 and 189/257 SCT tracks get an id. The rest have none.
  - It fixes two SCT id switches: cam01 track 223 is 554 then 553, and cam03 track 86 is 643 then 644.
    So the mapping is not strictly one-to-one. Prepare splits these tracks (see Ground truth runs).
  - Around that switch, cam01 track 225 (frames 4075–4161) has no Multi_CAM id but inherits 553.
    So 553 is on two cam01 tracks at once (data quality issue D20, still open).
  - cam02 id 431 (frames 1–52) has no SCT box. All world coordinates are `-1` (unknown).
- `Vehicle_statistic.xlsx`: one sheet per camera, one row per SCT track id (colour, make, model, category).
- `RoundaboutHD_Reid_subset/` (`test_images/` is the gallery):
  - train: 41,126 crops, 511 IDs
  - query: 1,171 crops, 310 IDs
  - gallery: 23,225 crops, 310 IDs
  - file names: `{id:04d}_cam0X_{frame:06d}.jpg`, with the frame 0-based (the README says 6-digit ids)
  - Crops are SCT boxes cut from the video. ReID ids are a separate numbering:
    the 310 query/gallery IDs map to Multi_CAM ids (99.7% of crops agree; 3 Multi_CAM ids have two ReID IDs).
    The 201 train-only IDs are single-camera vehicles with no Multi_CAM id.
  - All 310 query/gallery IDs also appear in train, from the same tracklets at interleaved frames (192 identical images).
    6 of them are seen by only one camera. Check the paper's protocol before reporting ReID scores.
