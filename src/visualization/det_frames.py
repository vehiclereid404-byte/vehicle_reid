"""Check images for the extracted detection frames and their labels (det_split and frames stages).

- reports/figures/step3_<split>.jpg: 2 frames per camera with the boxes read back from the YOLO label files,
  coloured by the state in the COCO file (green moving_sct, orange moving_bg, blue parked). Also checks that
  the YOLO boxes equal the COCO boxes and the COCO boxes equal outputs/detections/gt.
- reports/figures/step3_alignment.jpg: per camera, the fastest gt track box in a test frame, cut from the
  extracted frame f, with the gt boxes of frames f-1 (red), f (green) and f+1 (blue). Green must fit.
- Prints the mean absolute pixel difference between ReID crops and the same SCT box cut from the extracted
  frame f and from frame f + frame_step: low at f confirms the frame numbering of the extracted files.
Run from the repo root after the frames stage:  python -m src.visualization.det_frames
"""
import json

import cv2
import numpy as np
import pandas as pd

from src.common.config import ROOT, load_params
from src.common.io import DET_COLS, TRACK_COLS, read_table
from src.data.det_split import COCO_DIR, SPLITS
from src.data.frames import FRAME_DIR, YOLO_CLASS

OUT = ROOT / "reports" / "figures"
COLOURS = {"moving_sct": (0, 200, 0), "moving_bg": (0, 140, 255), "parked": (255, 120, 0)}  # BGR
COCO_ID = {i: k for k, i in YOLO_CLASS.items()}


def read_yolo(path, width: int, height: int) -> np.ndarray:
    """(N, 5): COCO class id, x, y, w, h in pixels."""
    rows = [list(map(float, line.split())) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return np.zeros((0, 5))
    a = np.array(rows)
    w, h = a[:, 3] * width, a[:, 4] * height
    return np.stack([[COCO_ID[int(k)] for k in a[:, 0]], a[:, 1] * width - w / 2, a[:, 2] * height - h / 2, w, h], 1)


def check_labels(split: str, cameras: list[str]) -> dict:
    """Largest corner difference (px) YOLO vs COCO and COCO vs gt detections, over all images of a split."""
    coco = json.loads((COCO_DIR / f"{split}.json").read_text(encoding="utf-8"))
    anns = pd.DataFrame(coco["annotations"])
    worst_yolo, worst_gt, class_errors = 0.0, 0.0, 0
    gt = {c: read_table(ROOT / "outputs" / "detections" / "gt" / f"{c}.csv", DET_COLS) for c in cameras}
    by_image = dict(tuple(anns.groupby("image_id")))
    for im in coco["images"]:
        a = by_image.get(im["id"], anns.iloc[:0])
        box = np.array(a.bbox.tolist()).reshape(-1, 4)
        y = read_yolo((FRAME_DIR / im["file_name"]).with_suffix(".txt"), im["width"], im["height"])
        g = gt[im["cam"]][gt[im["cam"]].frame == im["frame"]]
        if len(y) != len(a) or len(g) != len(a):
            raise AssertionError(f"{im['file_name']}: {len(a)} COCO, {len(y)} YOLO, {len(g)} gt boxes")
        if len(a):
            worst_yolo = max(worst_yolo, float(np.abs(y[:, 1:] - box).max()))
            worst_gt = max(worst_gt, float(np.abs(g[["x", "y", "w", "h"]].to_numpy() - box).max()))
            class_errors += int((y[:, 0] != a.category_id.to_numpy()).sum() + (g.cls.to_numpy() != a.category_id).sum())
    return dict(images=len(coco["images"]), boxes=len(anns), yolo_vs_coco_px=round(worst_yolo, 3),
                coco_vs_gt_px=round(worst_gt, 3), class_errors=class_errors)


def panel(im: dict, anns: pd.DataFrame, width: int = 960) -> np.ndarray:
    """One frame, resized to `width`, with its YOLO boxes coloured by COCO state."""
    image = cv2.imread(str(FRAME_DIR / im["file_name"]))
    scale = width / image.shape[1]
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    yolo = read_yolo((FRAME_DIR / im["file_name"]).with_suffix(".txt"), im["width"], im["height"])
    for (k, x, y, w, h), state in zip(yolo, anns.state):
        p1, p2 = (int(x * scale), int(y * scale)), (int((x + w) * scale), int((y + h) * scale))
        cv2.rectangle(image, p1, p2, COLOURS[state], 1 if state == "parked" else 2)
        if k != 2:
            cv2.putText(image, {3: "moto", 5: "bus", 7: "truck"}[int(k)], (p1[0], p1[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOURS[state], 1, cv2.LINE_AA)
    text = f"{im['cam']} frame {im['frame']}: {len(anns)} boxes"
    cv2.rectangle(image, (0, 0), (300, 22), (0, 0, 0), -1)
    cv2.putText(image, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def split_figure(split: str, cameras: list[str]) -> np.ndarray:
    """4 cameras x 2 frames (at 1/3 and 2/3 of the split's detection frames)."""
    coco = json.loads((COCO_DIR / f"{split}.json").read_text(encoding="utf-8"))
    anns = pd.DataFrame(coco["annotations"])
    rows = []
    for cam in cameras:
        ims = [im for im in coco["images"] if im["cam"] == cam]
        picks = [ims[len(ims) // 3], ims[2 * len(ims) // 3]]
        rows.append(np.hstack([panel(im, anns[anns.image_id == im["id"]]) for im in picks]))
    legend = np.zeros((26, rows[0].shape[1], 3), np.uint8)
    cv2.putText(legend, f"{split}: gt boxes from the YOLO labels; green moving (SCT), orange moving background, "
                "blue parked", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([legend, *rows])


def alignment_figure(cameras: list[str], step: int) -> np.ndarray:
    """Per camera: fastest gt track box in a test detection frame, with boxes of f-1, f, f+1."""
    test = json.loads((COCO_DIR / "test.json").read_text(encoding="utf-8"))
    tiles = []
    for cam in cameras:
        frames = {im["frame"] for im in test["images"] if im["cam"] == cam}
        trk = read_table(ROOT / "outputs" / "tracks" / "gt" / f"{cam}.csv", TRACK_COLS)
        trk = trk.assign(cx=trk.x + trk.w / 2, cy=trk.y + trk.h / 2).set_index(["track_id", "frame"])
        cur = trk[trk.index.get_level_values("frame").isin(frames)].reset_index()
        prev = trk.reindex(list(zip(cur.track_id, cur.frame - 1)))
        nxt = trk.reindex(list(zip(cur.track_id, cur.frame + 1)))
        speed = np.hypot(nxt.cx.to_numpy() - prev.cx.to_numpy(), nxt.cy.to_numpy() - prev.cy.to_numpy()) / 2
        speed = np.where(cur.w.to_numpy() > 150, speed, 0)  # a box large enough to see the fit
        i = int(np.nanargmax(speed))
        row, f = cur.iloc[i], int(cur.frame.iloc[i])
        image = cv2.imread(str(FRAME_DIR / cam / f"{f:06d}.jpg"))
        m = int(max(row.w, row.h) * 0.6)
        x0, y0 = max(int(row.x) - m, 0), max(int(row.y) - m, 0)
        x1, y1 = min(int(row.x + row.w) + m, image.shape[1]), min(int(row.y + row.h) + m, image.shape[0])
        crop = image[y0:y1, x0:x1].copy()
        for box, colour in [(prev.iloc[i], (0, 0, 255)), (nxt.iloc[i], (255, 0, 0)), (row, (0, 220, 0))]:
            cv2.rectangle(crop, (int(box.x) - x0, int(box.y) - y0), (int(box.x + box.w) - x0, int(box.y + box.h) - y0),
                          colour, 3)
        scale = 420 / crop.shape[0]
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.putText(crop, f"{cam} f={f} track {int(row.track_id)}, {speed[i]:.0f} px/frame", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(crop)
    legend = np.zeros((26, sum(t.shape[1] for t in tiles), 3), np.uint8)
    cv2.putText(legend, "extracted frame f with gt track boxes of f-1 (red), f (green), f+1 (blue)", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([legend, np.hstack(tiles)])


def crop_check(cameras: list[str], step: int, per_cam: int = 150) -> pd.DataFrame:
    """Mean absolute difference (0-255) of ReID crops vs the same SCT box cut from extracted frame f and f + step."""
    crops = pd.read_csv(ROOT / "outputs" / "reid_split" / "gt" / "crops.csv")
    index = pd.read_csv(FRAME_DIR / "index.csv")
    have = set(zip(index.cam, index.frame))
    rows = []
    for cam in cameras:
        trk = read_table(ROOT / "outputs" / "tracks" / "gt" / f"{cam}.csv", TRACK_COLS).set_index(["track_id", "frame"])
        c = crops[(crops.cam == cam) & np.array([(cam, f) in have and (cam, f + step) in have for f in crops.frame])]
        c = c[np.array([(t, f) in trk.index for t, f in zip(c.track_id, c.frame)], dtype=bool)]
        c = c.iloc[np.linspace(0, len(c) - 1, min(per_cam, len(c))).astype(int)]
        for r in c.itertuples(index=False):
            ref = cv2.imread(str(ROOT / r.path)).astype(np.float32)
            b = trk.loc[(r.track_id, r.frame)]
            x1, y1 = int(b.x), int(b.y)
            mae = []
            for f in (r.frame, r.frame + step):
                cut = cv2.imread(str(FRAME_DIR / cam / f"{f:06d}.jpg"))[y1:y1 + ref.shape[0], x1:x1 + ref.shape[1]]
                mae.append(float(np.abs(cut.astype(np.float32) - ref[:cut.shape[0], :cut.shape[1]]).mean())
                           if cut.shape[:2] == ref.shape[:2] else np.nan)
            rows.append(dict(cam=cam, mae_f=mae[0], mae_f_plus_step=mae[1]))
    return pd.DataFrame(rows)


def main() -> None:
    params = load_params()
    cameras, step = params["cameras"], params["det_split"]["frame_step"]
    OUT.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        print(split, check_labels(split, cameras))
        path = OUT / f"step3_{split}.jpg"
        cv2.imwrite(str(path), split_figure(split, cameras), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(path)
    path = OUT / "step3_alignment.jpg"
    cv2.imwrite(str(path), alignment_figure(cameras, step), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(path)
    mae = crop_check(cameras, step)
    print("ReID crop vs extracted frame, mean abs. difference (median over crops):")
    print(mae.groupby("cam")[["mae_f", "mae_f_plus_step"]].median().round(2).assign(
        crops=mae.groupby("cam").size()).to_string())


if __name__ == "__main__":
    main()
