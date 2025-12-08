# utils.py
from __future__ import annotations
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import json


def set_seed(seed: int = 42, deterministic: bool = True):
    """Set seeds across common libraries and enable deterministic flags."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it does not already exist."""

    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_experiment_report(
    output_dir: str | Path,
    config: Dict[str, Any],
    metrics: Dict[str, float],
    notes: Optional[str] = None,
) -> Path:
    """Persist a JSON experiment report for reproducibility."""

    report_dir = ensure_dir(output_dir)
    payload: Dict[str, Any] = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "config": config,
        "metrics": metrics,
    }
    if notes:
        payload["notes"] = notes

    report_path = report_dir / "experiment_summary.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return report_path
