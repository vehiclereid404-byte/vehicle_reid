"""Read camera frames straight from the raw videos."""
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def video_path(raw_dir: str | Path, cam: str) -> Path:
    """cam01 -> <raw_dir>/imagesc001/video.mp4"""
    return Path(raw_dir) / f"imagesc0{cam[-2:]}" / "video.mp4"


def iter_frames(path: str | Path, start: int = 1, stop: int | None = None,
                step: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_number, BGR image). Frame numbers are 1-based, like SCT_GT.txt."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start - 1)
    n = start
    try:
        while stop is None or n <= stop:
            if (n - start) % step == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                yield n, frame
            elif not cap.grab():
                break
            n += 1
    finally:
        cap.release()
