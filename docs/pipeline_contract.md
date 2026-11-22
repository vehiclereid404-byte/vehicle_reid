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
                                   outputs/eval_masks/    (tracking scoring masks)
RoundaboutHD_Reid_subset ─ reid_split ─► outputs/reid_split/gt/   (ID-disjoint ReID train/val/test split)
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
| `outputs/eval_masks/skip_frames.csv` | `cam,frame`: frames dropped from tracking scoring | prepare |
| `outputs/eval_masks/ignore_regions.json` | `{cam: [polygon, ...]}`, polygon = `[[x, y], ...]` in full-frame pixels | prepare |
| `outputs/reid_split/gt/crops.csv` | `path,cam,frame,track_id,global_id,vehicle_id,group,file_id,official,split,role,w,h`, one row per ReID crop (see ReID split) | reid_split |

Read and write these with `src/common/io.py`, so the column checks stay in one place.

## Ground truth runs (`gt`, written by `python -m src.data.prepare`)

- `outputs/detections/gt/<cam>.csv`: every `labels_GT` box, including parked vehicles.
  Clipped to the frame, `score` = 1.0, `cls` as labelled (2, 3, 5, 7).
- `outputs/tracks/gt/<cam>.csv`: `SCT_GT` clipped to the frame, `score` = 1.0, `cls` = 2.
  For a duplicate `(frame, track_id)`, the first row is kept.
- **Multi_CAM corrections** (`prepare.mc_corrections` in `params.yaml`, `src/data/prepare.py:correct_multicam`):
  Multi_CAM rows that are wrong on the video are relabelled or dropped right after reading, before they are matched
  to SCT tracks. Each correction names the camera, id, frame range, new id (`null` = drop), issue code and reason;
  `reports/data_quality.json` logs the rows changed.
  - D06: cam01 553, frames 4075–4188 → 554 (114 rows). The boxes are track 223, the white pickup (554).
  - D24: cam03 553, frames 339–599 dropped (261 rows). The boxes are track 29, a street sweeper, which so falls back to
    the single-camera id 30029.
- **ID switches:** Multi_CAM sometimes gives one SCT track two vehicle ids in separate time spans.
  Such a track is split at the first frame of the later id, and the later part gets track id = max track id on that camera + 1.
  Here (after the corrections) there is one: cam03 86 → 526 from frame 1288 (D07).
- **Lost ids (D20, `src/data/prepare.py:split_lost_ids`):** a Multi_CAM id moves from track A to track B of the
  same camera at frame f, and all of A's rows at frames ≥ f have no Multi_CAM id while B carries it.
  Then A is split at f: the later part gets track id = max track id on that camera + 1, and so its own
  single-camera global id. Only cases where A and B would otherwise share a global id in the same frame are split.
  After the Multi_CAM corrections it finds no case; it stays as a check.
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
    The ignore region (below) now drops most of them from tracking scoring.
- **Tracking scoring masks** (`outputs/eval_masks/`, written by prepare). The tracking scorer applies them with
  `src/evaluation/masks.py:apply_masks(df, cam)` to gt and predicted rows alike, after dropping parked tracks.
  Detection scoring doesn't use them.
  - **Skip frames (D11):** frames with no SCT rows (cam02 2946, cam04 4099–4229) are dropped.
  - **Ignore region (D23):** one region per camera, a list of polygons around the area SCT annotates.
    A row is dropped if its box bottom-centre `(x + w/2, y + h)` is outside every polygon (on the edge counts as inside).
    - The region is built from the data (`src/data/regions.py`, `region_*` in `params.yaml`): mark the bottom edges
      of all gt track boxes on an 8-px grid, close by 64 px, dilate by 16 px and simplify by 8 px.
    - It keeps 100% of the gt track rows and covers 47–62% of the image.
    - It drops 73%, 79%, 59% and 26% (cam01..cam04) of the moving `labels_GT` boxes that have no SCT box.
      cam04 drops the fewest: most of those boxes are on its far approach road, inside the stretch where SCT annotates some vehicles but not others.
    - Check images: `reports/figures/ignore_region_cam0N.jpg` (`python -m src.visualization.ignore_regions`).
- `reports/data_quality.json`: a list of `{id, title, count, per_camera, fix_applied}` with stable ids
  (`D01`, `D02`, ...). `python -m src.data.checks` prints it.

## ReID split (`reid_split`, written by `python -m src.data.reid_split`)

An ID-disjoint train/val/test split of all ReID crops (official train, query and test folders together),
for training and scoring ReID models. Settings: `reid_split` and `seed` in `params.yaml`.

- `outputs/reid_split/gt/crops.csv`, one row per unique image (byte-identical copies are dropped: 650 of 65,522 files):
  - `path` (repo-relative, forward slashes), `cam`, `frame` (1-based SCT frame = file-name frame + 1),
    `track_id` (gt track, after the D07 split), `global_id` (from `outputs/matching/gt/global_ids.csv`).
  - `vehicle_id`: the identity to train and score on: `global_id` with the identity fixes below.
  - `group`: split group (smallest `vehicle_id` in it). Vehicles that may be one physical vehicle share a group.
  - `file_id`: the id in the file name, a separate numbering (never equal to `global_id`).
  - `official`: the official folder: `train`, `query`, `gallery` (= `test_images`), or `query+gallery` /
    `train+query` for byte-identical copies. Use it to score the paper's protocol.
  - `split` (`train`, `val`, `test`), `role` (`train`, `query`, `gallery`), `w`, `h` (crop size in pixels).
  - LF line endings. Also written: `outputs/reid_split/gt/checks.json` (file-id conflicts, grouped pairs) and
    `reports/metrics/reid_split_gt.json` (counts).
- **Linking:** each crop is `img[int(y1):int(y2), int(x1):int(x2)]` of an SCT box at frame (file-name frame + 1),
  checked on the video pixels. Its size equals that box's integer size for 64,871 of 64,872 crops (one is 1 px off).
  22 crops fit two same-size boxes in their frame; the track that the file id's other crops link to is used (checked on the video).
- **Identity fixes** (in `src/data/reid_split.py`, checked on the crops, `reports/figures/reid_check_*.jpg`; not in prepare yet):
  - Multi_CAM 161 → 78: one black Chevrolet Colorado (cam01 78, cam02–03 161).
  - cam03 track 526 and cam04 track 81 (both Multi_CAM 644) → 87: the tan Kia Sorento, which is 87 on cam02.
    The rest of 644 (cam01, cam02, cam03 track 78) is a red dump truck.
- **Split groups:** two vehicles join a group if
  - crops of both on one camera differ in ≤ `near_dup_bits` (24) of 256 dHash bits (near-duplicates; this finds the same
    vehicle on two trips, see Dataset facts, and also some look-alikes), or
  - a single-camera vehicle may be a missed Multi_CAM link of a multi-camera one: same make and model in
    `Vehicle_statistic.xlsx`, on a neighbouring camera the multi-camera vehicle is not seen on, ≤ `link_window_frames` (1,500) apart.
  - Vehicles in a group of 2+ stay in train (17 + 25 pairs → 33 groups, 75 vehicles), so no near-duplicate or
    possible same vehicle crosses splits and none is scored under two labels.
- **Eligible for val/test:** a Multi_CAM vehicle (`vehicle_id` < 10000) with crops on ≥ 2 cameras, ≥ 2 crops on at least one
  camera, alone in its group (263 of 310). Eligible vehicles are shuffled (`seed`) within each number of cameras (2, 3, 4):
  40% test, 15% val, the rest train with every other vehicle.
- **Query and gallery:** in val and test, for each vehicle and each camera with ≥ 2 crops, one random (seeded) crop is the
  query; all other crops of the split are gallery. There are no distractor vehicles.
- **Protocol:** each query ranks the gallery of its own split. Gallery crops with the same `vehicle_id` and the same `cam`
  as the query are ignored. Report mAP and CMC Rank-1/5/10 over all queries. Every query has a cross-camera match (checked).
- **Counts (seed 42):**

  | split | vehicles | query | gallery or train crops |
  |---|---|---|---|
  | train | 366 (166 Multi_CAM, 200 single-camera) | – | 39,417 |
  | val | 39 | 105 | 6,953 |
  | test | 105 | 284 | 18,113 |

- **Crop size vs the 256×128 Hailo ReID input:** the median crop is 160 × 101 px (w × h); w/h has median 1.56
  (10–90%: 1.1–2.4). 64% of crops are under 128 px tall and 75% under 256 px wide, so most are upscaled at any of these sizes.
  256×128 as H × W (portrait, the person-ReID shape of Hailo's RepVGG-A0 model) squeezes a median vehicle's width about 3×
  relative to its height. W × H = 256 × 128 (landscape, same pixel count and cost) fits the vehicles better;
  256 × 256 doubles the cost. The reid stage picks one and scores fp32 and int8 at that size.

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
| `data-eval` | `src/data/`, `src/evaluation/`, `src/visualization/`, `docs/licenses.md`, DVC stages `prepare` and `reid_split` |
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
  The videos are not in DVC (`.dvcignore`); everyone copies them from the shared drive.
  `raw_data.dvc` versions the rest of `data/raw` (1.2 GB). A DVC stage can still list a video as a dep.
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
  - Frames with no rows: cam02 2946, cam04 4099–4229. Tracking scoring skips them (D11).
- `Multi_CAM_Ground_Turth.txt` (the misspelling is in the file name):
  `cam id frame x y w h xworld yworld`.
  - Boxes and frames are SCT_GT's: 99.97% of rows equal an SCT box in the same frame.
    `id` is a cross-camera vehicle id, unrelated to SCT track ids, so map it per camera by box match.
  - 311 ids, 305 of them seen by 2–4 cameras. Per camera, 179/262, 221/244, 245/279 and 189/257 SCT tracks get an id
    (cam03: 244/279 after the D24 correction). The rest have none.
  - It fixes one SCT id switch: cam03 track 86 is 643 (frames 1134–1139, a red SUV) then 644 (1288+, a grey SUV).
    So the mapping is not strictly one-to-one. Prepare splits that track (see Ground truth runs).
  - It has two errors of its own, checked on the video and corrected by prepare (see Multi_CAM corrections):
    - cam01 (D06): tracks 223 (white service pickup, 554) and 225 (silver Subaru, 553) pass behind the "MARMON"
      sign around frames 4070–4076. After it, Multi_CAM puts 553 on track 223's boxes (frames 4075–4188), and
      track 225 (frames 4075–4161) has no id. The SCT tracks are right. Before the correction this also put 553 on
      two cam01 tracks at once (D20).
    - cam03 (D24): 553 is reused for two vehicles, track 29 (frames 339–599, a street sweeper) and track 293
      (frames 5378–5689, the Subaru). It is the only Multi_CAM id that spans more than 1,800 frames on one camera.
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
  - Byte-identical copies: 458 query images are also in `test_images` and 192 in `train_images` (same file name = same bytes).
    8 official queries (6 IDs) have no gallery image from another camera.
  - File ids follow the Multi_CAM ids in order but drift from them (never equal); single-camera vehicles are 754–954.
    Linked to the corrected identities (see ReID split):
    - file 673 is two vehicles: a white semi (469, cam01–02) and the box truck 664 (cam03–04).
      The box truck is also files 25 (cam01) and 852 (cam02).
    - file 953 is two cam04 vehicles: a red Toyota Yaris (track 364, also file 954) and a black sedan (track 365).
    - files 592 (78 + 161) and 653 (87 + part of 644) are right; Multi_CAM is wrong there.
  - **Multi_CAM ids are per trip, not per vehicle:** a vehicle that passes a camera twice gets two ids. Seen on the
    crops: black Silverado 666 / 685 (cam02), a liveried blue Nissan 691 / 10171 (cam01); an IMI concrete mixer 83 / 653 (cam03)
    is likely the same truck. The ReID file ids are per trip too. So one physical vehicle can have two `global_id`s;
    the ReID split keeps such candidates together in train (`reports/figures/reid_check_repeat_visits.jpg`).
