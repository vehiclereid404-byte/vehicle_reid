"""Small box helpers shared by prepare and checks. Boxes are x1, y1, x2, y2 columns."""
import numpy as np
import pandas as pd

XYXY = ["x1", "y1", "x2", "y2"]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (N, 4) and b (M, 4), both x1y1x2y2."""
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_by_frame(a: pd.DataFrame, b: pd.DataFrame, min_iou: float) -> tuple[np.ndarray, np.ndarray]:
    """For each row of `a`, find the best-IoU row of `b` in the same frame.

    Returns (index label of that row in b, or -1 if IoU < min_iou; best IoU), aligned with a's rows.
    """
    idx = np.full(len(a), -1, dtype=np.int64)
    best = np.zeros(len(a))
    boxes_a, boxes_b = a[XYXY].to_numpy(float), b[XYXY].to_numpy(float)
    labels_b = b.index.to_numpy()
    groups_b = b.groupby("frame").indices
    for frame, rows_a in a.groupby("frame").indices.items():
        rows_b = groups_b.get(frame)
        if rows_b is None:
            continue
        iou = iou_matrix(boxes_a[rows_a], boxes_b[rows_b])
        j = iou.argmax(1)
        best[rows_a] = iou[np.arange(len(rows_a)), j]
        ok = best[rows_a] >= min_iou
        idx[rows_a[ok]] = labels_b[rows_b[j[ok]]]
    return idx, best


def outside_frame(df: pd.DataFrame, width: int, height: int) -> pd.Series:
    """True for boxes that reach past the frame edge."""
    return (df.x1 < 0) | (df.y1 < 0) | (df.x2 > width) | (df.y2 > height)


def clip_to_xywh(df: pd.DataFrame, width: int, height: int) -> pd.DataFrame:
    """Clip x1y1x2y2 to the frame and add x, y, w, h columns (rounded to 0.01 px)."""
    out = df.copy()
    x1, x2 = out.x1.clip(0, width), out.x2.clip(0, width)
    y1, y2 = out.y1.clip(0, height), out.y2.clip(0, height)
    out["x"], out["y"] = x1.round(2), y1.round(2)
    out["w"], out["h"] = (x2 - x1).round(2), (y2 - y1).round(2)
    return out
