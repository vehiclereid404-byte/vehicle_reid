# Data

RoundaboutHD (MIT licence): 4 cameras around a roundabout, 4K, 9,000 frames each. Data files are **not in Git**. They are stored on
DagsHub (`https://dagshub.com/Udhay/vehicle_reid`) and handled with DVC.

```
data/
├── raw/RoundaboutHD/            ← original dataset (DVC: raw_data.dvc), never edit
│   ├── RoundaboutHD_Reid_subset/   ReID crops (train/query/test_images)
│   ├── imagesc00x/                 per camera: video.mp4, SCT/ (tracking GT), detection/ (label zips), geo-mapping/
│   ├── Multi_CAM_Ground_Turth.txt  cross-camera GT
│   └── Vehicle_statistic.xlsx      type/colour/make/model per vehicle
└── processed/reid/crops.csv     ← our clean ReID split (DVC: dvc.lock, stage process_data)
```

## 1. Get the data

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# DagsHub → Settings → Tokens. --local keeps the token out of Git (.dvc/config.local)
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <dagshub-username>
dvc remote modify origin --local password <token>
```

Pull only what you need. The four `video.mp4` files are 6.4 GB of the 8.1 GB and are only needed for detection/tracking:

```bash
# ReID (≈1.2 GB): crops + annotations + our split
dvc pull data/raw/RoundaboutHD/RoundaboutHD_Reid_subset
dvc pull data/raw/RoundaboutHD/Multi_CAM_Ground_Turth.txt
dvc pull data/raw/RoundaboutHD/Vehicle_statistic.xlsx
for c in 1 2 3 4; do dvc pull data/raw/RoundaboutHD/imagesc00$c/SCT; done
dvc pull process_data                      # data/processed/reid/crops.csv

# Detection / tracking: add labels and videos
for c in 1 2 3 4; do
  dvc pull data/raw/RoundaboutHD/imagesc00$c/detection
  dvc pull data/raw/RoundaboutHD/imagesc00$c/video.mp4
done
```

Pull **one path per command**. Passing many paths inside `data/raw` in a single `dvc pull` silently fetched nothing for us.

> ⚠️ With a partial pull, DVC reports `data/raw` as modified. That's expected. **Never run `dvc add data/raw`, `dvc commit`,
> or a plain `dvc repro`**: each of these re-records `data/raw` without the missing files and would overwrite the team's
> dataset version. Use `dvc repro -s process_data` (see 3.).

## 2. Use `crops.csv` for ReID training

One row per crop. Images are not copied; `path` points into `data/raw` (relative to the repo root).

| column | meaning |
|---|---|
| `path` | image file |
| `id` | vehicle ID (**the label**) |
| `cam` | camera 1–4 |
| `frame` | frame number (0-based, = SCT frame − 1) |
| `split` | `train` / `val` / `test`, **disjoint by vehicle ID** |
| `role` | `train`, or `query` / `gallery` in val and test |
| `official_split` | split of the original dataset (leaky, only for comparison with the paper) |
| `track_id`, `gid` | per-camera SCT track ID, global cross-camera ID (empty if the vehicle isn't in the multi-camera GT) |
| `w`, `h` | crop size in px |
| `category`, `colour`, `make`, `model` | vehicle attributes (for error analysis) |

| split | IDs | crops |
|---|---|---|
| train | 358 | 36,996 |
| val | 51 | 137 query / 9,378 gallery |
| test | 102 | 279 query / 18,081 gallery |

```python
import pandas as pd
df = pd.read_csv("data/processed/reid/crops.csv")
train = df[df.split == "train"]
label = {vid: i for i, vid in enumerate(sorted(train.id.unique()))}   # id -> class index for cross-entropy
val_q, val_g = df[(df.split == "val") & (df.role == "query")], df[(df.split == "val") & (df.role == "gallery")]
```

- **Train** on `split == "train"` (PK sampler groups by `id`).
- **Tune** on val: rank the val gallery for each val query and compute mAP / Rank-1.
  **Evaluation protocol:** for each query, ignore gallery crops with the same `id` *and* the same `cam` (only cross-camera matches count).
- **Test** only for final numbers. Don't tune on it.
- Crop sizes vary a lot (median 160×101, max 2183×1737) → resize to a fixed input size in the transform.

## 3. Rebuild `crops.csv`

```bash
dvc repro -s process_data   # runs src/data/make_dataset.py if its inputs changed, updates dvc.lock
dvc push                    # upload the new version, then commit dvc.lock
```

## 4. What we did and why

The original data has several problems. All of them are listed in [`reports/data_issues.md`](../reports/data_issues.md).
The main ones for ReID:

1. **The official ReID split leaks.** It was split by image, not by vehicle: all 310 test vehicles also appear in train
   (interleaved frames of the same tracks), and 650 query images are exact copies of gallery or train images.
   → We re-split **by vehicle ID** (seed 42): 1/3 of the multi-camera vehicles for test, 1/6 for val, the rest train.
   Val/test only use vehicles seen by ≥ 2 cameras, so every query has a cross-camera match.
2. **File-name IDs are their own numbering** (not the global IDs of the multi-camera GT).
   → Each crop is linked to its SCT track and global ID by (camera, frame + 1, crop size), 99.9% linked (`src/data/load.py:link_reid`).
3. **ID 673 contains two different trucks** → its camera 3–4 crops got the new ID 955.
4. Duplicates removed, IDs with < 2 crops dropped.

Notebooks (run top to bottom from `notebooks/`):
- `1.0-…-raw-inventory-and-annotations`: all annotation files: formats, README mismatches, label errors.
- `2.0-…-reid-subset-checks`: ReID crops: leakage, duplicates, ID links, visual checks.
- `3.0-…-prepare-reid-data`: step-by-step version of `src/data/make_dataset.py` (the script is the source of truth).

Important for **detection/tracking**: there are no frame images in the dataset, so frames have to be extracted from `video.mp4`.
The detection label file `imgN.txt` belongs to SCT frame **N+1**.
