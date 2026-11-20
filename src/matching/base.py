"""Interface every cross-camera matcher implements. One matcher per file in src/matching/."""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Matcher(ABC):
    @abstractmethod
    def match(self, tracks: dict[str, pd.DataFrame],
              embeddings: dict[str, tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
        """Per-camera tracks (TRACK_COLS) and (track_ids, emb) -> DataFrame cam, track_id, global_id."""
