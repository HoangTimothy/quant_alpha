"""
Shared helper utilities for the quant_alpha project.
Logging, seeding, IO, device management, timing.
"""

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ── Seeding ───────────────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    """Set random seed for full reproducibility across numpy, torch, python."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ── Device ────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    """Return the best available torch device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── IO ────────────────────────────────────────────────────────────────
def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist. Returns Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: dict | list, path: str | Path) -> None:
    """Write data to a JSON file with pretty-printing."""
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_serializer)


def load_json(path: str | Path) -> Any:
    """Read and return data from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_serializer(obj: Any) -> Any:
    """Fallback serializer for numpy/torch types inside JSON dumps."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ── Logging ───────────────────────────────────────────────────────────
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ── Timing ────────────────────────────────────────────────────────────
@contextmanager
def timer(label: str = "Block", logger: logging.Logger | None = None):
    """Context manager that logs elapsed wall-clock time."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    msg = f"⏱  {label} completed in {elapsed:.2f}s"
    if logger:
        logger.info(msg)
    else:
        print(msg)
