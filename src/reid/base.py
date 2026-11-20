"""Interface every ReID model implements. One model per file in src/reid/."""
from abc import ABC, abstractmethod

import numpy as np


class ReIDModel(ABC):
    @abstractmethod
    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """BGR vehicle crops -> (N, D) float32 embeddings, L2-normalised."""
