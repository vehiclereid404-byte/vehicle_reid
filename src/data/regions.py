"""Ignore regions for tracking scoring (data quality issue D23).

SCT_GT annotates only the roundabout and the lanes next to it. Moving cars on background
roads have labels_GT boxes but no tracks, so every tracker would be charged false
positives for them. A camera's region is the area where SCT_GT vehicles touch the road:

1. mark the bottom edge (x .. x+w at y+h) of every gt track box on a grid of `cell_px` cells,
2. close the gaps between neighbouring lanes (`close_px`) and dilate by `margin_px`,
3. keep the outer contours, simplified with cv2.approxPolyDP (`simplify_px`),
4. store them in full-resolution pixels.

A box is inside if its bottom-centre (x + w/2, y + h) is inside or on any polygon.
prepare writes the regions and the skipped frames to MASK_DIR; scorers use
src/evaluation/masks.py:apply_masks.
"""
import cv2
import numpy as np
import pandas as pd

from src.common.config import ROOT

MASK_DIR = ROOT / "outputs" / "eval_masks"
SKIP_FILE = "skip_frames.csv"         # cam,frame: frames dropped from tracking scoring (D11)
REGION_FILE = "ignore_regions.json"   # {cam: [polygon, ...]}, polygon = [[x, y], ...] in pixels (D23)


def bottom_centre(df: pd.DataFrame) -> np.ndarray:
    """(N, 2) test points of boxes in x, y, w, h: the middle of the bottom edge."""
    return np.stack([df.x + df.w / 2, df.y + df.h], axis=1).astype(float)


def bottom_edge_mask(tracks: pd.DataFrame, width: int, height: int, cell_px: int) -> np.ndarray:
    """uint8 grid of cell_px cells, 1 where the bottom edge of any box lies."""
    ny, nx = -(-height // cell_px), -(-width // cell_px)
    row = ((tracks.y + tracks.h).to_numpy() // cell_px).astype(int).clip(0, ny - 1)
    col1 = (tracks.x.to_numpy() // cell_px).astype(int).clip(0, nx - 1)
    col2 = ((tracks.x + tracks.w).to_numpy() // cell_px).astype(int).clip(0, nx - 1)
    diff = np.zeros((ny, nx + 1), np.int32)  # +1 where an edge starts, -1 after it ends
    np.add.at(diff, (row, col1), 1)
    np.add.at(diff, (row, col2 + 1), -1)
    return (diff.cumsum(axis=1)[:, :nx] > 0).astype(np.uint8)


def region_polygons(tracks: pd.DataFrame, width: int, height: int, cell_px: int, close_px: float,
                    margin_px: float, simplify_px: float) -> list[list[list[int]]]:
    """Polygons around the bottom edges of all boxes in a tracks table (x, y, w, h)."""
    def disk(px: float) -> np.ndarray:
        r = max(1, round(px / cell_px))
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

    mask = bottom_edge_mask(tracks, width, height, cell_px)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disk(close_px))
    mask = cv2.dilate(mask, disk(margin_px))
    ny, nx = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        simple = cv2.approxPolyDP(contour, simplify_px / cell_px, True)
        c = (simple if len(simple) >= 3 else contour)[:, 0, :]
        # cell index -> pixel at the cell centre; cells on the grid border go to the frame edge
        x = np.where(c[:, 0] == 0, 0, np.where(c[:, 0] == nx - 1, width, (c[:, 0] + 0.5) * cell_px))
        y = np.where(c[:, 1] == 0, 0, np.where(c[:, 1] == ny - 1, height, (c[:, 1] + 0.5) * cell_px))
        polygons.append(np.stack([x, y], axis=1).round().astype(int).tolist())
    return polygons


def inside_region(df: pd.DataFrame, polygons: list) -> np.ndarray:
    """True for boxes (x, y, w, h) whose bottom-centre is inside or on one of the polygons."""
    points = bottom_centre(df)
    inside = np.zeros(len(df), dtype=bool)
    for poly in polygons:
        contour = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
        inside |= np.array([cv2.pointPolygonTest(contour, (x, y), False) >= 0 for x, y in points.tolist()],
                           dtype=bool)
    return inside


def area_fraction(polygons: list, width: int, height: int) -> float:
    """Share of the frame covered by the polygons (outer contours, so they don't overlap)."""
    return sum(cv2.contourArea(np.asarray(p, dtype=np.float32)) for p in polygons) / (width * height)
