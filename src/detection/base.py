"""Interface every detector implements. One detector per file in src/detection/."""
from abc import ABC, abstractmethod

import numpy as np


class Detector(ABC):
    @abstractmethod
    def detect(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """BGR frames -> one (N, 6) array per frame: x, y, w, h, score, cls.

        Boxes are absolute pixels, (x, y) is the top-left corner, cls uses COCO ids.
        """
