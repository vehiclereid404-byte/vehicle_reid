"""Interface every tracker implements. One tracker per file in src/tracking/."""
from abc import ABC, abstractmethod

import numpy as np


class Tracker(ABC):
    @abstractmethod
    def reset(self) -> None:
        """Forget all tracks; called before each camera's video."""

    @abstractmethod
    def update(self, dets: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """(N, 6) detections for one frame -> (M, 7) tracks: track_id, x, y, w, h, score, cls.

        `frame` is the BGR image, for trackers that use appearance.
        """
