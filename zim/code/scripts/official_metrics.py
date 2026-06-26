from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image


def read_alpha(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    img = Image.open(path).convert("L")
    if size and img.size != size:
        img = img.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def save_alpha(path: str | Path, alpha: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(alpha, 0, 1) * 255 + 0.5).astype(np.uint8), mode="L").save(path)


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | str]:
    pred = np.clip(pred.astype(np.float32), 0, 1)
    gt = np.clip(gt.astype(np.float32), 0, 1)
    if pred.shape != gt.shape:
        pred_img = Image.fromarray((pred * 255 + 0.5).astype(np.uint8), mode="L")
        pred = np.asarray(pred_img.resize((gt.shape[1], gt.shape[0]), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    diff = np.abs(pred - gt)
    raw_sad = float(diff.sum())
    sad = raw_sad / 1000.0
    mse = float(np.mean((pred - gt) ** 2))
    gy_p, gx_p = np.gradient(pred)
    gy_g, gx_g = np.gradient(gt)
    grad = float(np.abs(np.sqrt(gx_p * gx_p + gy_p * gy_p) - np.sqrt(gx_g * gx_g + gy_g * gy_g)).sum() / 1000.0)
    boundary = (gt > 0.05) & (gt < 0.95)
    if boundary.any():
        boundary_sad = float(diff[boundary].sum() / 1000.0)
        boundary_mse = float(np.mean((pred[boundary] - gt[boundary]) ** 2))
    else:
        boundary_sad = math.nan
        boundary_mse = math.nan
    connectivity = float(np.logical_xor(pred >= 0.5, gt >= 0.5).sum() / 1000.0)
    return {
        "SAD": sad,
        "raw_SAD": raw_sad,
        "MSE": mse,
        "Gradient": grad,
        "Connectivity": connectivity,
        "Boundary_SAD": boundary_sad,
        "Boundary_MSE": boundary_mse,
        "metric_notes": "Gradient/connectivity are local approximations unless official eval script is used.",
    }
