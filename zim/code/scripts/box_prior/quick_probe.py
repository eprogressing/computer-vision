#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path("/home/lpy/anisorisk/computer_vison")
PREV_FINAL_RUN = ROOT / "runs/protocol_repair_final_claim_20260602_174358"
PREV_SPLIT_RUN = ROOT / "runs/promptmatte_lhr_20260604_010706"
CURRENT = "zim_vitb_flip_tta_bbox_guided_r1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_alpha(path: Path | str, size: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("L")
    if img.size != size:
        img = img.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def parse_bbox(row: dict[str, str]) -> list[int]:
    return [int(round(float(x))) for x in json.loads(row["bbox"])]


def expand_bbox(bbox: list[int], width: int, height: int, pad: float) -> list[int]:
    x1, y1, x2, y2 = [float(x) for x in bbox]
    bw = max(1.0, x2 - x1 + 1)
    bh = max(1.0, y2 - y1 + 1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return [
        int(max(0, math.floor(cx - bw * pad / 2))),
        int(max(0, math.floor(cy - bh * pad / 2))),
        int(min(width - 1, math.ceil(cx + bw * pad / 2))),
        int(min(height - 1, math.ceil(cy + bh * pad / 2))),
    ]


def support(size: tuple[int, int], bbox: list[int], pad: float, blur: float) -> np.ndarray:
    w, h = size
    x1, y1, x2, y2 = expand_bbox(bbox, w, h, pad)
    arr = np.zeros((h, w), dtype=np.uint8)
    arr[y1 : y2 + 1, x1 : x2 + 1] = 255
    img = Image.fromarray(arr, mode="L")
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))
    return np.asarray(img, dtype=np.float32) / 255.0


def gt_boundary_band(gt: np.ndarray) -> np.ndarray:
    fg = gt >= 0.5
    edge = np.zeros_like(fg, dtype=bool)
    edge[1:, :] |= fg[1:, :] != fg[:-1, :]
    edge[:-1, :] |= fg[:-1, :] != fg[1:, :]
    edge[:, 1:] |= fg[:, 1:] != fg[:, :-1]
    edge[:, :-1] |= fg[:, :-1] != fg[:, 1:]
    return edge | ((gt > 0.02) & (gt < 0.98))


def metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    diff = np.abs(pred - gt)
    b = gt_boundary_band(gt)
    return {
        "SAD": float(diff.sum() / 1000),
        "MSE": float(np.mean((pred - gt) ** 2)),
        "Boundary_SAD": float(diff[b].sum() / 1000) if b.any() else math.nan,
    }


def main() -> None:
    prev = read_csv(PREV_FINAL_RUN / "metrics/final_metrics_all.csv")
    amap = {(r["sample_id"], r["method"]): Path(r["alpha_path_pred"]) for r in prev}
    params = [
        ("bp_p1.5_s0.05_f0.2", 1.5, 8.0, 0.05, 0.2),
        ("bp_p1.5_s0.10_f0.2", 1.5, 8.0, 0.10, 0.2),
        ("bp_p1.75_s0.05_f0.2", 1.75, 10.0, 0.05, 0.2),
        ("bp_p1.75_s0.10_f0.2", 1.75, 10.0, 0.10, 0.2),
        ("bp_p2.0_s0.05_f0.2", 2.0, 12.0, 0.05, 0.2),
        ("bp_p2.0_s0.10_f0.2", 2.0, 12.0, 0.10, 0.2),
        ("bp_p2.5_s0.05_f0.3", 2.5, 14.0, 0.05, 0.3),
        ("bp_p2.5_s0.10_f0.3", 2.5, 14.0, 0.10, 0.3),
    ]
    for split in ["dev_lhr100", "val_lhr100"]:
        rows = {CURRENT: []}
        rows.update({p[0]: [] for p in params})
        for row in read_csv(PREV_SPLIT_RUN / "manifests" / f"{split}.csv"):
            sid = row["sample_id"]
            size = Image.open(row["image_path"]).size
            cur = read_alpha(amap[(sid, CURRENT)], size)
            gt = read_alpha(row["alpha_path"], size)
            rows[CURRENT].append(metrics(cur, gt))
            bbox = parse_bbox(row)
            for name, pad, blur, suppress, floor in params:
                m = support(size, bbox, pad, blur)
                gate = m + (1 - m) * floor
                pred = np.clip(cur * (1 - suppress + suppress * gate), 0, 1)
                rows[name].append(metrics(pred, gt))
        print("SPLIT", split)
        curm = {k: float(np.mean([r[k] for r in rows[CURRENT]])) for k in ["SAD", "MSE", "Boundary_SAD"]}
        print("CURRENT", curm)
        for name, *_ in params:
            mm = {k: float(np.mean([r[k] for r in rows[name]])) for k in ["SAD", "MSE", "Boundary_SAD"]}
            print(name, mm, {k + "_imp": (curm[k] - mm[k]) / curm[k] for k in curm})


if __name__ == "__main__":
    main()
