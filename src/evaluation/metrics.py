"""Write a scorer's result as a flat {"metric": number} JSON in reports/metrics/ (docs/pipeline_contract.md)."""
import json
from pathlib import Path

from src.common.config import ROOT

METRICS_DIR = ROOT / "reports" / "metrics"


def save_metrics(metrics: dict, name: str) -> Path:
    """reports/metrics/<name>.json, e.g. name = "det_yolov9_test". Floats are rounded to 4 decimals."""
    flat = {k: round(float(v), 4) if isinstance(v, float) else v for k, v in metrics.items()}
    path = METRICS_DIR / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(flat, indent=1) + "\n", encoding="utf-8", newline="\n")
    return path
