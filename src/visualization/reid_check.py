"""Side-by-side crops for the ReID identity checks -> reports/figures/reid_check_*.jpg.

One row per (camera, track): up to 6 crops spread over the track, with the frame on each crop.
Reads outputs/reid_split/gt/crops.csv.  Run:  python -m src.visualization.reid_check
"""
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.common.config import ROOT

FIG_DIR = ROOT / "reports" / "figures"
ROW_H, PER_ROW, LABEL_W = 150, 6, 330

# figure name -> rows (cam, track_id); see docs/pipeline_contract.md "ReID split"
CHECKS = {
    "78_161": [("cam01", 426), ("cam02", 356), ("cam03", 504)],  # one black Chevrolet Colorado
    "87_644": [("cam03", 78), ("cam02", 97), ("cam01", 162),     # 644: red dump truck
               ("cam02", 28), ("cam03", 526), ("cam04", 81)],   # 87 on cam02, 644 on cam03/04: tan Kia Sorento
    "673": [("cam02", 133), ("cam01", 236),                     # file id 673 part 1: white semi (469)
            ("cam01", 159), ("cam02", 163), ("cam03", 259), ("cam04", 229)],  # 664: box truck (file ids 25, 852, 673)
    "repeat_visits": [("cam02", 197), ("cam02", 271),           # black Silverado, Multi_CAM 666 and 685
                      ("cam01", 353), ("cam01", 171),           # liveried blue Nissan, 691 and single-camera 10171
                      ("cam03", 69), ("cam03", 167)],           # IMI concrete mixer, 83 and 653
}


def track_row(crops: pd.DataFrame, cam: str, track: int) -> Image.Image:
    sel = crops[(crops.cam == cam) & (crops.track_id == track)].sort_values("frame")
    pick = sel.iloc[np.linspace(0, len(sel) - 1, min(PER_ROW, len(sel))).round().astype(int)]
    r = sel.iloc[0]
    label = [f"{cam} track {track}", f"global_id {r.global_id}  vehicle_id {r.vehicle_id}",
             f"file id {sorted(set(sel.file_id))}", f"frames {sel.frame.min()}-{sel.frame.max()}"]
    ims = []
    for path, frame in zip(pick.path, pick.frame):
        im = Image.open(ROOT / path).convert("RGB")
        im = im.resize((max(1, round(im.width * ROW_H / im.height)), ROW_H))
        ImageDraw.Draw(im).text((3, 3), str(frame), fill=(255, 255, 0))
        ims.append(im)
    row = Image.new("RGB", (LABEL_W + sum(i.width + 4 for i in ims), ROW_H), (30, 30, 30))
    ImageDraw.Draw(row).multiline_text((6, 40), "\n".join(label), fill=(255, 255, 255), spacing=6)
    x = LABEL_W
    for im in ims:
        row.paste(im, (x, 0))
        x += im.width + 4
    return row


def main() -> None:
    crops = pd.read_csv(ROOT / "outputs" / "reid_split" / "gt" / "crops.csv")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for name, tracks in CHECKS.items():
        rows = [track_row(crops, cam, track) for cam, track in tracks]
        fig = Image.new("RGB", (max(r.width for r in rows), sum(r.height + 6 for r in rows)))
        y = 0
        for r in rows:
            fig.paste(r, (0, y))
            y += r.height + 6
        out = FIG_DIR / f"reid_check_{name}.jpg"
        fig.save(out, quality=85)
        print(out.relative_to(ROOT).as_posix(), fig.size)


if __name__ == "__main__":
    main()
