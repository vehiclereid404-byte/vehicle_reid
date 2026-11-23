"""Sanity tests of src/evaluation/det_eval.py on the test split (needs `dvc repro det_split`).

Run from the repo root:  python -m pytest tests/test_det_eval.py -q
"""
import json

import numpy as np
import pandas as pd
import pytest

from src.common.config import load_params
from src.data.det_split import COCO_DIR
from src.data.regions import inside_region
from src.evaluation.det_eval import read_predictions, region_gt, score_detections
from src.evaluation.masks import load_masks

SPLIT = "test"
pytestmark = pytest.mark.skipif(not (COCO_DIR / f"{SPLIT}_1cls.json").exists(), reason="run dvc repro det_split")


def score(preds: pd.DataFrame) -> dict:
    return score_detections(preds.reset_index(drop=True), SPLIT, by_camera=False)


@pytest.fixture(scope="module")
def gt() -> pd.DataFrame:
    """The gt run on the split's detection frames."""
    coco = json.loads((COCO_DIR / f"{SPLIT}_1cls.json").read_text(encoding="utf-8"))
    images = pd.DataFrame(coco["images"])[["cam", "frame"]]
    return read_predictions("gt", load_params()["cameras"]).merge(images, on=["cam", "frame"])


@pytest.fixture(scope="module")
def boxes() -> pd.DataFrame:
    """gt boxes of the split with cam, frame and target (moving and inside the region)."""
    coco = json.loads((COCO_DIR / f"{SPLIT}_1cls.json").read_text(encoding="utf-8"))
    d, _ = region_gt(coco, load_masks()[1])
    im = {i["id"]: (i["cam"], i["frame"]) for i in d["images"]}
    return pd.DataFrame([dict(cam=im[a["image_id"]][0], frame=im[a["image_id"]][1], x=a["bbox"][0], y=a["bbox"][1],
                              w=a["bbox"][2], h=a["bbox"][3], score=1.0, cls=2, target=not a["iscrowd"])
                         for a in d["annotations"]])


@pytest.fixture(scope="module")
def base(gt) -> dict:
    return score(gt)


def test_gt_vs_gt_is_perfect(base):
    for s in ["all", "moving_in_region"]:
        for k in ["AP", "AP50", "AP75", "AR"]:
            assert base[f"{s}.{k}"] == pytest.approx(1.0), (s, k)


def test_dropping_10_percent_lowers_both(gt, base):
    kept = gt.sample(frac=0.9, random_state=0)
    m = score(kept)
    assert m["all.AR"] == pytest.approx(0.9, abs=0.01)
    assert 0.85 < m["all.AP"] < 0.95
    assert m["moving_in_region.AR"] < 0.95 and m["moving_in_region.AP"] < 0.95


def test_shift_lowers_strict_thresholds_only(gt):
    m = score(gt.assign(x=gt.x + 0.1 * gt.w))  # IoU 0.9 / 1.1 = 0.82: a hit at IoU .50-.80, a miss at .85-.95
    assert m["all.AP50"] == pytest.approx(1.0) and m["all.AP75"] == pytest.approx(1.0)
    assert m["all.AP"] == pytest.approx(0.7, abs=0.02)
    assert m["moving_in_region.AP"] == pytest.approx(0.7, abs=0.03)


def test_detection_on_ignored_parked_car_changes_nothing(boxes):
    targets = boxes[boxes.target].assign(score=0.5)
    regions = load_masks()[1]
    parked = boxes[~boxes.target]
    parked = parked[np.concatenate([inside_region(g, regions[c]) for c, g in parked.groupby("cam")])]
    assert len(parked) > 0  # parked boxes inside the region (35 in test)

    def shrunk(f):  # same bottom-centre (so still inside the region), IoU with the parked box = f^2
        return parked.assign(x=parked.x + parked.w * (1 - f) / 2, y=parked.y + parked.h * (1 - f),
                             w=parked.w * f, h=parked.h * f, score=1.0)

    ref = score(targets)
    m = score(pd.concat([targets, shrunk(0.8)]))  # IoU 0.64 on an ignore box, scored above every target
    assert ref["moving_in_region.AP"] == pytest.approx(1.0)
    assert {k: v for k, v in m.items() if k.startswith("moving_in_region.") and k != "moving_in_region.dets_dropped_outside"} \
        == {k: v for k, v in ref.items() if k.startswith("moving_in_region.") and k != "moving_in_region.dets_dropped_outside"}
    control = score(pd.concat([targets, shrunk(0.6)]))  # IoU 0.36: not on the parked car -> false positives
    assert control["moving_in_region.AP"] < 1.0


def test_detection_outside_region_is_dropped(gt, boxes, base):
    regions = load_masks()[1]
    extra = []
    for cam, g in boxes.groupby("cam"):  # one 80 x 50 box per image, outside the region and clear of every gt box
        xs, ys = np.meshgrid(np.arange(100, 3800, 100), np.arange(100, 2100, 100))
        cand = pd.DataFrame(dict(x=xs.ravel() - 40.0, y=ys.ravel() - 50.0, w=80.0, h=50.0))
        cand = cand[~inside_region(cand, regions[cam])]
        clear = [((g.x < r.x + r.w) & (r.x < g.x + g.w) & (g.y < r.y + r.h) & (r.y < g.y + g.h)).sum() == 0
                 for r in cand.itertuples()]
        r = cand[clear].iloc[0]
        frames = gt.frame[gt.cam == cam].unique()
        extra.append(pd.DataFrame(dict(cam=cam, frame=frames, x=r.x, y=r.y, w=r.w, h=r.h, score=1.0, cls=2)))
    extra = pd.concat(extra)
    m = score(pd.concat([gt.assign(score=0.5), extra]))
    assert m["moving_in_region.dets_dropped_outside"] == base["moving_in_region.dets_dropped_outside"] + len(extra)
    assert m["moving_in_region.AP"] == pytest.approx(1.0)
    assert m["all.AP"] < 1.0  # the same boxes are false positives when nothing is dropped
