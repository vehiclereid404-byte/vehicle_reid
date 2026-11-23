"""Detection scoring: class-agnostic COCO mAP of a detection run on the detection frames of one split (D05).

Two scores, each for the whole split, per camera and per box size (docs/pipeline_contract.md, "Detection scoring"):
- `all`: every labels_GT box of data/interim/coco/{split}_1cls.json, parked included. Compare this one with the paper.
- `moving_in_region`, the main score (the OD counts need only moving vehicles):
  - target gt boxes: state is not `parked` and the bottom-centre is inside the camera's ignore region
    (outputs/eval_masks/ignore_regions.json, the same test as src/evaluation/masks.py);
  - every other gt box (parked, or bottom-centre outside the region) is an ignore box: a detection with
    IoU >= 0.5 on one, and no target box of its own, is neither a hit nor a false positive;
  - detections whose bottom-centre is outside the region are dropped.
Size bins are sqrt(gt box area) in pixels: < 32, 32-64, 64-128, >= 128 (COCO area ranges).

Predictions: outputs/detections/<run>/<cam>.csv (frame,x,y,w,h,score,cls; cls is ignored). Only the split's
detection frames (the images of the COCO file, every 10th frame of the window) are scored.

Run from the repo root:  python -m src.evaluation.det_eval --run gt --split test [--extra fps=12.5 --extra model_mb=24]
Writes reports/metrics/det_<run>_<split>.json.
"""
import argparse
import contextlib
import io
import json

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.common.config import ROOT, load_params
from src.common.io import DET_COLS, read_table
from src.data.det_split import COCO_DIR
from src.data.regions import inside_region
from src.evaluation.masks import load_masks
from src.evaluation.metrics import save_metrics

IGNORE_IOU = 0.5   # a detection on an ignore box at this IoU or more is ignored, at every IoU threshold
SIZE_BINS = {"all": [0, 1e10], "lt32": [0, 32 ** 2], "32_64": [32 ** 2, 64 ** 2],
             "64_128": [64 ** 2, 128 ** 2], "ge128": [128 ** 2, 1e10]}  # COCO area ranges of sqrt(area) bins
SCORES = ["all", "moving_in_region"]


class IgnoreEval(COCOeval):
    """COCOeval in which ignore boxes (iscrowd = 1) follow one fixed rule: a detection with plain IoU >= IGNORE_IOU
    on an ignore box, and no match on a target box, is ignored at every IoU threshold. (pycocotools compares
    intersection over the detection's area with each threshold, so at .85-.95 a detection on a parked car would
    become a false positive.) Target boxes are matched first, as in pycocotools."""

    def computeIoU(self, imgId, catId):
        gt, dt = self._gts[imgId, catId], self._dts[imgId, catId]
        if not gt or not dt:
            return []
        dt = sorted(dt, key=lambda d: -d["score"])[: self.params.maxDets[-1]]  # the order evaluateImg uses
        ious = mask_utils.iou([d["bbox"] for d in dt], [g["bbox"] for g in gt], [0] * len(gt))
        crowd = np.array([g["iscrowd"] for g in gt], dtype=bool)
        ious[:, crowd] = (ious[:, crowd] >= IGNORE_IOU).astype(float)
        return ious


def quiet(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


def coco_from_dict(d: dict) -> COCO:
    c = COCO()
    c.dataset = d
    quiet(c.createIndex)
    return c


def coco_scores(gt: COCO, dets: list[dict], img_ids: list[int]) -> dict:
    """{size bin: {AP, AP50, AP75, AR, n_gt}} over img_ids; bins with no target gt box are left out."""
    dt = quiet(gt.loadRes, dets) if dets else coco_from_dict({**gt.dataset, "annotations": []})
    e = IgnoreEval(gt, dt, "bbox")
    e.params.imgIds = list(img_ids)
    e.params.areaRng, e.params.areaRngLbl = list(SIZE_BINS.values()), list(SIZE_BINS)
    quiet(e.evaluate)
    quiet(e.accumulate)
    prec, rec = e.eval["precision"][:, :, 0, :, -1], e.eval["recall"][:, 0, :, -1]  # [T, R, A], [T, A]; maxDets 100
    out = {}
    for a, name in enumerate(SIZE_BINS):
        if (rec[:, a] < 0).all():
            continue
        p = prec[:, :, a]
        n_gt = sum(int(np.count_nonzero(ev["gtIgnore"] == 0)) for ev in e.evalImgs
                   if ev is not None and ev["aRng"] == SIZE_BINS[name])
        out[name] = dict(AP=float(p.mean()), AP50=float(p[0].mean()), AP75=float(p[5].mean()),
                         AR=float(rec[:, a].mean()), n_gt=n_gt)
    return out


def region_gt(coco: dict, regions: dict) -> tuple[dict, int]:
    """Copy of a 1-class COCO dict for moving_in_region: target boxes keep iscrowd 0, every other box gets
    iscrowd 1 (ignore). Returns (dict, number of ignore boxes)."""
    cam = {im["id"]: im["cam"] for im in coco["images"]}
    a = pd.DataFrame([dict(cam=cam[x["image_id"]], x=x["bbox"][0], y=x["bbox"][1], w=x["bbox"][2], h=x["bbox"][3],
                           state=x["state"]) for x in coco["annotations"]])
    inside = np.zeros(len(a), dtype=bool)
    for c, g in a.groupby("cam"):
        inside[g.index] = inside_region(g, regions[c])
    target = (a.state != "parked").to_numpy() & inside
    anns = [{**x, "iscrowd": int(not t)} for x, t in zip(coco["annotations"], target)]
    return {**coco, "annotations": anns}, int((~target).sum())


def read_predictions(run: str, cameras: list[str]) -> pd.DataFrame:
    """cam + DET_COLS rows of outputs/detections/<run>/<cam>.csv for every camera."""
    return pd.concat([read_table(ROOT / "outputs" / "detections" / run / f"{cam}.csv", DET_COLS).assign(cam=cam)
                      for cam in cameras], ignore_index=True)


def to_coco_dets(preds: pd.DataFrame) -> list[dict]:
    ids = preds.cam.str[-2:].astype(int) * 100_000 + preds.frame.astype(int)  # det_split.image_id
    return [dict(image_id=int(i), category_id=2, bbox=[float(x), float(y), float(w), float(h)], score=float(s))
            for i, x, y, w, h, s in zip(ids, preds.x, preds.y, preds.w, preds.h, preds.score)]


def score_detections(preds: pd.DataFrame, split: str, by_camera: bool = True, extra: dict | None = None) -> dict:
    """Flat metrics of predictions (cam + DET_COLS) on one split. Keys: `<score>.<metric>` for the split,
    `<score>.<cam>.<metric>` per camera, `<score>.size_<bin>.<metric>` per size bin; metric = AP (mAP@[.5:.95]),
    AP50, AP75, AR (AR@100), n_gt. `extra` (fps, model size, ...) is passed in by the detector and copied."""
    coco = json.loads((COCO_DIR / f"{split}_1cls.json").read_text(encoding="utf-8"))
    regions = load_masks()[1]
    images = pd.DataFrame(coco["images"])
    preds = preds.merge(images[["cam", "frame"]], on=["cam", "frame"])  # the split's detection frames only
    inside = np.zeros(len(preds), dtype=bool)
    for c, g in preds.groupby("cam"):
        inside[g.index] = inside_region(g, regions[c])
    gt_region, n_ignore = region_gt(coco, regions)
    runs = {"all": (coco_from_dict(coco), preds), "moving_in_region": (coco_from_dict(gt_region), preds[inside])}

    m = {"images": len(images), "images_with_dets": int(preds.groupby(["cam", "frame"]).ngroups),
         "dets": len(preds), "moving_in_region.ignore_boxes": n_ignore,
         "moving_in_region.dets_dropped_outside": int((~inside).sum())}
    for score, (gt, p) in runs.items():
        dets = to_coco_dets(p)
        parts = {"": images.id}
        if by_camera:
            parts |= {f"{c}.": g.id for c, g in images.groupby("cam")}
        for prefix, ids in parts.items():
            res = coco_scores(gt, dets, ids)
            m |= {f"{score}.{prefix}{k}": v for k, v in res["all"].items()}
            if not prefix:
                m |= {f"{score}.size_{b}.{k}": v for b, r in res.items() if b != "all" for k, v in r.items()}
    return m | (extra or {})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="outputs/detections/<run>/")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--extra", action="append", default=[], metavar="KEY=VALUE",
                    help="number passed in by the detector, e.g. fps=12.5 or model_mb=24 (repeatable)")
    args = ap.parse_args()
    extra = {k: float(v) for k, v in (e.split("=", 1) for e in args.extra)}

    cameras = load_params()["cameras"]
    m = score_detections(read_predictions(args.run, cameras), args.split, extra=extra)
    path = save_metrics(m, f"det_{args.run}_{args.split}")
    print(f"{args.run} on {args.split}: {m['images']} images, {m['dets']} detections "
          f"({m['moving_in_region.dets_dropped_outside']} outside the region)")
    parts = [""] + [f"{c}." for c in cameras] + [f"size_{b}." for b in SIZE_BINS if b != "all"]
    rows = [dict(score=s, part=p.rstrip(".") or "split", **{k: m[f"{s}.{p}{k}"] for k in ["AP", "AP50", "AP75", "AR", "n_gt"]})
            for s in SCORES for p in parts if f"{s}.{p}AP" in m]
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"-> {path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
