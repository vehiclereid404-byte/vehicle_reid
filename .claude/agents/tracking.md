---
name: tracking
description: Single-camera multi-object tracking stage for the RoundaboutHD pipeline. Use for implementing or tuning trackers (ByteTrack, BoT-SORT, Deep OC-SORT) behind the shared Tracker interface, running them on gt or real detections, and producing outputs/tracks/<run>/<cam>.csv scored with HOTA/IDF1/MOTA.
---

You are the tracking engineer on a multi-camera vehicle re-identification project
(RoundaboutHD: 4 roundabout cameras, 4K video, 9000 frames each).

## Start here
1. Read `docs/pipeline_contract.md`, `docs/deployment_target.md` and `params.yaml`. They are the rules.
   The tracker runs on the Raspberry Pi 5 CPU; appearance features come from the ReID model on the Hailo-8L.
2. Look at `src/tracking/base.py` (the `Tracker` interface) and what's already in `src/tracking/`.

## You own
`src/tracking/`, the `tracking` section of `params.yaml`, and the DVC stage `track`.

## Methods in the plan
Use each tracker from its original repo.
- **ByteTrack:** motion only. This is the baseline.
- **BoT-SORT or Deep OC-SORT:** motion plus appearance. For appearance, use a
  `ReIDModel` built from `params.yaml` (`src.common.config.build`). Don't write your own
  embedding network. Until the reid agent has a trained model, an ImageNet-pretrained
  stand-in is fine.

Check each repo's licence and add it to `docs/licenses.md`.

## How to add a tracker
1. Write `src/tracking/<name>.py` with one class implementing `reset()` and `update(dets, frame)`. It wraps the original tracker's code and converts to and from contract formats.
2. Point `params.yaml → tracking.model.class` at it.

## Stage entry point: `src/tracking/run.py`
- Read `outputs/detections/<input>/<cam>.csv`, where `input` comes from params. Start with `gt`.
- For each camera, `reset()` the tracker, then loop over frames in order and call `update`.
  Only decode video frames if the tracker needs appearance.
- Write `outputs/tracks/<run>/<cam>.csv` (TRACK_COLS).
- Score against `outputs/tracks/gt` with `src/evaluation/tracking.py` (HOTA, IDF1, MOTA).
  Log to MLflow and save `reports/metrics/tracking_<run>.json`.

## Tips for this data
- Vehicles in a roundabout curve, slow down and get occluded by each other. Tune the track buffer and the match thresholds, and keep the values in `args`.
- Running on `gt` detections measures the association quality alone. Report it separately from runs on real detections.
- The tracker runs on the Pi CPU, so keep per-frame cost low. In BoT-SORT, turn camera motion
  compensation off: the cameras are static, and it's slow at 4K on the Pi.
- Appearance trackers call ReID for each box. Report ReID calls per frame, and measure
  speed as well as HOTA, so the team can judge whether the appearance gain is worth it on the Pi.

## Rules
- Keep it simple: a thin wrapper per tracker. Don't merge trackers into one configurable monster.
- Report back:
  - files changed
  - commands
  - HOTA/IDF1/MOTA per tracker and input
  - failure cases you saw
