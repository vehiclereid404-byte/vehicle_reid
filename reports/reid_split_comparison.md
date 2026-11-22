# ReID split: ours vs Simon's

- **Ours:** `outputs/reid_split/gt/crops.csv`, built by `src/data/reid_split.py`, DVC stage `reid_split`.
  The format and protocol are in `docs/pipeline_contract.md` ("ReID split").
- **Simon's:** `data/processed/reid/crops.csv`, built by `src/data/make_dataset.py`, DVC stage `process_data`.

Both splits are ID-disjoint and drop byte-identical copies. Both put only multi-camera vehicles in val and test, use one query per
vehicle and camera, and ignore same-ID, same-camera gallery crops. They differ in what counts as "one vehicle".

## Side by side

| | Ours | Simon's |
|---|---|---|
| Split by | `vehicle_id`: prepare's `global_id` (with D06, D24 and D07), plus 161 → 78 and the Kia part of 644 → 87 | file-name id, with 673 split into 673 and 955 |
| Train: IDs / crops | 366 / 39,417 | 358 / 36,996 |
| Val: IDs / queries / gallery | 39 / 105 / 6,953 | 51 / 137 / 9,378 |
| Test: IDs / queries / gallery | 105 / 284 / 18,113 | 102 / 279 / 18,081 |
| Crops in total | 64,872 (every unique image) | 64,871 (drops file 939, which has 1 crop) |
| Their val/test vehicles in the other split's train | our val: 20 of 39 in his train; our test: 54 of 105 | his val: 23 of 51 in our train; his test: 59 of 102 |
| Real vehicle on two sides (corrected ids) | 0 (checked by assert) | **1**: box truck 664: test as file 955 (cam03–04), train as files 25 (cam01) and 852 (cam02) |
| Same vehicle on two trips, on two sides | 0 | **1** confirmed: black Silverado 666 (test, file 675) and 685 (train, file 694) |
| Possible same vehicle on two sides | 0 (33 groups with 75 vehicles stay in train) | 20 of our groups, including the two above: 10 near-duplicate pairs and 13 missed-link pairs |
| 78 + 161 (one Colorado) | one vehicle (78), train | one file id (592), test. Correct |
| 87 + 644 | Kia (87) and dump truck (644) are separate, both in test | file 653 (Kia) and file 606 (truck), both in test. Correct |
| Byte-identical copies | one row; `official` keeps both folders (`query+gallery`, `train+query`) | one row; `official_split` keeps one folder (`query`) |
| Near-duplicates across splits | vehicle pairs with crops within 24 of 256 dHash bits on one camera are grouped and kept in train; 0 cross splits | not checked |
| Eligible for val/test | Multi_CAM vehicle on 2+ cameras, 2+ crops on one camera, not grouped: 263 of 310 | every file id on 2+ cameras |
| Queries | 1 seeded random crop per vehicle and camera with 2+ crops; the rest is gallery | 1 random crop per id and camera; the rest is gallery |
| Fractions, seed | 40% test, 15% val of eligible vehicles; seed 42 | 1/3 test, 1/6 val of multi-camera ids; seed 42 |
| Crop → SCT track | 64,872 of 64,872 (100%) by exact integer box size; the 22 same-size ties are resolved by the file id's track | 64,827 of 64,871 (99.93%) by float size within 3 px; 44 unlinked, and **22 on the wrong track** (the same-size ties) |
| Track ids | gt tracks; cam03 86 is split into 86 and 526 (D07) | raw SCT; the 77 crops of track 526 stay on 86 |
| Frame | 1-based SCT frame (contract) | 0-based file-name frame |
| `gid` column | `global_id` from prepare; the identity used is `vehicle_id` | his own match: D06 has no effect (majority vote); **D24: the 99 street-sweeper crops have 553 (the Subaru)**; the Kia's cam03/04 crops have 644; 161 not merged with 78; 40 crops have no gid (unlinked) and 21 a wrong one (wrong-track ties). The gid isn't used to split, so his split is unaffected. |
| Extra columns | `global_id`, `vehicle_id`, `group`, `file_id` | `category`, `colour`, `make`, `model` |

We share only 31 test vehicles. Both splits are random draws, so test scores from the two splits are not comparable.

## Simon's findings, checked

| Finding | Verdict | Numbers |
|---|---|---|
| File ids differ from global ids | **True** | 0 of 64,872 crops have file id = global id (or = SCT track id). File ids follow the Multi_CAM order with a drift: only 24 of 311 are global id + 1. Single-camera vehicles are 754–954. |
| ID 673 is two vehicles (semi on cams 1–2, box truck on cams 3–4) | **True** | 313 crops of the white semi (469: cam02 track 133, cam01 track 236) and 191 of the box truck (664: cam03 259, cam04 229). The box truck's cam01 and cam02 crops are files 25 and 852. His fix doesn't join them, which causes the leak above. `figures/reid_check_673.jpg` |
| 458 query images are byte-identical to gallery images | **True** | 458 (same name and same md5; md5 and name map 1:1) |
| 192 query images are identical to train images | **True** | 192 |
| 8 queries (6 IDs) have no cross-camera gallery image | **True** | files 25, 73, 131, 185 (2 queries), 359 and 437 (2 queries). With corrected identities it's 7 queries (5 vehicles): file 25's box truck has gallery crops under file 673. |
| Global ids 78 + 161 are one vehicle | **True** | black Chevrolet Colorado: 78 on cam01 (frames 7276–7431), 161 on cam02 (7915–8033) and cam03 (8718–8856); file 592 covers all three. `figures/reid_check_78_161.jpg` |
| Global ids 87 + 644 are one vehicle | **Partly true** | 644 is two vehicles. One is a red dump truck: cam03 track 78 (1123–1279), cam02 97 and cam01 162 (file 606). The other is the tan Kia Sorento: cam03 track 526 (1288–1479) and cam04 track 81 (file 653). Only the Kia is 87 (cam02 track 28, 479–593). `figures/reid_check_87_644.jpg` |

## Other findings

- **Multi_CAM ids are per trip, not per vehicle.** A vehicle that passes a camera twice gets two ids, and the ReID file ids
  copy that. The near-duplicate check found them:
  - Clearly the same: the black Silverado 666/685 and a blue Nissan with "Molly Maid" livery, 691/10171. The IMI concrete mixer 83/653 is likely the same
    (`figures/reid_check_repeat_visits.jpg`).
  - Plausibly the same: Cruze 580/669, Escape 724/10251, Sonic 714/10498, Tiguan 269/40360, F-150 181/40001, RAV4 200/693 and Pilot 108/694.
  - Our split keeps all of them in train. Simon's has the Silverado on both sides, and 5 of the plausible pairs straddle his splits
    (108/694, 269/40360, 580/669, 714/10498, 724/10251).
- **File 953 is two vehicles:** a red Toyota Yaris (cam04 track 364, which is also file 954) and a black sedan (track 365).
  Both are single-camera and in train in both splits. Ours labels them by track, so they're right; in his, file 953 covers both (4 + 4 crops).
- **The cost of grouping:** 47 of the 310 multi-camera vehicles can't go to val or test (42 are grouped; 5 have crops on one
  camera only). Some groups are look-alikes rather than one vehicle (for example black sedans of different makes at the
  same spot). `near_dup_bits` and `link_window_frames` in `params.yaml` control this.
- Our val is smaller (39 IDs, 105 queries) than Simon's (51, 137). Raise `reid_split.val_frac` if model selection is noisy.

## Recommendation

**Use our split as the team's ReID split, and add Simon's attribute columns to it.**

- **Ours, because:**
  - It splits by the corrected physical vehicle. No confirmed vehicle is on two sides; Simon's has two (the box truck and the Silverado).
  - Possible same-vehicle pairs (repeat trips, missed Multi_CAM links) stay out of val and test, so test isn't scored
    with one vehicle under two labels.
  - All crops are linked, checked on the video. Simon's has 22 wrong-track and 44 unlinked crops.
  - It follows the contract: 1-based frames, gt track ids with D07, global ids with D06 and D24, and both official folders kept.
- **From Simon's:**
  - his `category, colour, make, model` columns, for per-category error analysis. They join on (cam, track_id); we add them to ours.
  - his notebooks and `reports/data_issues.md` as the inspection record. All his findings hold except "87 + 644", which is partly true.
- **Next:**
  - The reid agent switches to `outputs/reid_split/gt/crops.csv`. Then `process_data` can be retired (the team decides).
  - Once the team accepts 161 → 78 and the Kia → 87, move them into prepare and remove the mapping from `src/data/reid_split.py`.
  - Decide whether to merge the confirmed repeat trips (666/685, 691/10171, probably 83/653) into one id, for matching and OD too.

Reproduce: `dvc repro reid_split`, then `python -m src.visualization.reid_check` for the figures.
