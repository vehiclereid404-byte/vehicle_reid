"""Read and write the files stages pass to each other (see docs/pipeline_contract.md)."""
from pathlib import Path

import numpy as np
import pandas as pd

DET_COLS = ["frame", "x", "y", "w", "h", "score", "cls"]
TRACK_COLS = ["frame", "track_id", "x", "y", "w", "h", "score", "cls"]
GLOBAL_COLS = ["cam", "track_id", "global_id"]


def write_table(df: pd.DataFrame, path: str | Path, columns: list[str]) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[columns].to_csv(path, index=False)


def read_table(path: str | Path, columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return df[columns]


def save_embeddings(path: str | Path, track_ids: np.ndarray, emb: np.ndarray) -> None:
    if len(track_ids) != len(emb):
        raise ValueError("track_ids and emb must have the same length")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, track_id=np.asarray(track_ids), emb=np.asarray(emb, dtype=np.float32))


def load_embeddings(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["track_id"], data["emb"]
