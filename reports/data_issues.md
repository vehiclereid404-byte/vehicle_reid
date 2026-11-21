# RoundaboutHD – data issues

Found while inspecting the DagsHub copy of `data/raw/RoundaboutHD` (md5-identical to the shared drive copy). Details and code:
`notebooks/1.0-sh-raw-inventory-and-annotations.ipynb`, `notebooks/2.0-sh-reid-subset-checks.ipynb`.

## ReID subset (most important)
- **Official split leaks:** the split was made by image, not by vehicle ID. All 310 query/test vehicles also appear in `train_images`,
  as interleaved frames of the same tracks. Models trained on it have seen the test vehicles.
- **Duplicates:** 458 query images are byte-identical to gallery images, and 192 are identical to train images.
- **File-name IDs are their own numbering.** They are neither the Multi-CAM global IDs nor the SCT track IDs. Link via
  (cam, frame+1, crop size) → SCT track → global ID (`src/data/load.py:link_reid`).
- **ID 673 merges two vehicles** (Freightliner semi in cams 1–2, box truck in cams 3–4).
- 8 queries (6 IDs) have no gallery image from another camera.
- **Fix:** `data/processed/reid/crops.csv` (notebook 3) has an ID-disjoint train/val/test split (358/51/102 IDs) with our own
  query/gallery, duplicates removed, and ID 673 split (cams 3–4 → ID 955). The official split is kept in the `official_split` column.
  Evaluation: for each query, ignore gallery crops of the same ID from the same camera.

## Multi-camera GT (`Multi_CAM_Ground_Turth.txt`)
- **World coordinates are missing:** `xworld/yworld = -1` on 100% of rows. They would have to be computed from `geo-mapping/` if needed (OD matrix).
- Covers 834 of 1,042 SCT tracks (311 global IDs).
- **Missed cross-camera links:** e.g. global IDs 78 + 161 and 87 + 644 are the same vehicles (confirmed visually). Matters for evaluating cross-camera matching.
- 26 duplicate (cam, frame, id) rows (the same 26 as in SCT).

## SCT GT (`imagesc00x/SCT/SCT_GT.txt`)
- Box format is **x1 y1 x2 y2** (README doesn't say). Frames are 1-based.
- `track_id` is **per camera**, not global.
- `cls` is always 2 (car), even for trucks/buses. Use `Category` from `Vehicle_statistic.xlsx` (keyed by cam + track_id).
- 0.32% of boxes exceed the 3840×2160 frame by ≤ 11 px → clip.

## Detection labels (`imagesc00x/detection/*.zip`) – for the detection/tracking roles
- **Off-by-one:** `imgN.txt` = SCT frame **N+1** (and ReID crop frame N = SCT frame N+1). Watch this when extracting frames from the videos.
- Rows are `cls x1 y1 x2 y2 conf` in pixels with COCO class ids (2 car, 7 truck, 5 bus, 3 motorcycle) → convert for YOLO (normalised cx cy w h).
- `labels_GT` includes parked vehicles (~16 boxes/frame vs ~6.5 in SCT) and still has detector confidences (0.3–1.0).
  `labels_test` = SCT boxes. `labels_xy` = raw YOLOv12x output (not ground truth).
- No frame images in the dataset: frames have to be extracted from `video.mp4` (9,000 per camera).

## README vs. files
- Names differ: `Multi_CAM_Ground_Turth.txt` (typo), `Vehicle_statistic.xlsx`, `SCT/SCT_GT.txt`, ReID subset already unzipped as
  `RoundaboutHD_Reid_subset/`. The folders inside the detection zips have different names (`labels_filtered/`, `labels_sct2det/`, `labels_xy/`).
- `camXX_undistorted.png` is missing; `camXX.yaml` is undocumented.
- `Vehicle_statistic.xlsx`: only the first 6 columns are data; the rest is a category legend. 3 rows have no make/model.
