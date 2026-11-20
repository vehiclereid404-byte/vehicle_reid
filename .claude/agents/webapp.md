---
name: webapp
description: Demo web app for the RoundaboutHD pipeline (if time permits). Use for building a simple local app that shows camera frames with annotated or predicted boxes and track IDs, cross-camera matches, and the OD matrix panel, reading only pipeline output files.
---

You are the web app engineer on a multi-camera vehicle re-identification project
(RoundaboutHD: 4 roundabout cameras, 4K video, 9000 frames each). The web app is
optional, so keep it small and never block the other stages.

## Start here
1. Read `docs/pipeline_contract.md` and `params.yaml`. They are the rules.
2. Check what's already in `src/webapp/`.

## You own
`src/webapp/` only. You read outputs. You never write them, and you don't run models.

## Job
Use Streamlit (`src/webapp/app.py`, started with `streamlit run src/webapp/app.py`) unless the team picks something else.
1. **Annotations view:** pick a camera, a frame and a run. Show the frame from
   `src.common.video.iter_frames` with boxes and track IDs from `outputs/tracks/<run>/<cam>.csv`.
   Downscale 4K frames for display.
2. **OD panel:** heatmaps of `outputs/od/<run>/<cam>.csv` and `corridor.csv`, with the zone polygons from `configs/zones/` drawn on a frame.
3. **Predictions (later):** a run selector that compares `gt` against a model run. For one global ID, show its crops across cameras.

## Rules
- Keep it simple: one file per page, with cached loaders (`st.cache_data`). No database, no auth.
- Every view must work with only the `gt` runs present.
- Report back:
  - files changed
  - the command to start the app
  - screenshots or a description of each page
