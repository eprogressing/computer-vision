#!/usr/bin/env python3
"""
Metrics Computation Script
==========================
Computes SAD, MSE, MAE_x1000, and Boundary_SAD for predicted alpha
masks against GT alpha from the manifest.

Produces:
  - metrics_all.csv: per-sample rows
  - leaderboard.csv: aggregated per-method rows

Usage:
  python compute_metrics.py --manifest valid706_manifest_for_alignment.csv \\
      --predictions-dir outputs/ --methods sam2_bbox_binary,sam2_guided
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    MANIFEST_VALID706,
    OUTPUT_DIR,
    ALIGNMENT_PACKAGE,
    BOUNDARY_WIDTH,
    EXPECTED_N_VALID706,
    DATASET_BASE_OVERRIDE,
)


def fix_path(manifest_path_str: str) -> Path:
    """Convert manifest Linux path to local path if override is set."""
    p = Path(manifest_path_str)
    if DATASET_BASE_OVERRIDE is not None:
        parts = p.parts
        try:
            idx = parts.index("MicroMat3K_hf")
            relative = Path(*parts[idx + 1 :])
            return DATASET_BASE_OVERRIDE / relative
        except ValueError:
            pass
    return p


def load_alpha(path: Path) -> np.ndarray:
    """Load alpha image as float32 in [0, 1]."""
    img = Image.open(path).convert("L")
    alpha = np.array(img).astype(np.float32) / 255.0
    return alpha


def compute_sad(pred: np.ndarray, gt: np.ndarray) -> float:
    """Sum of Absolute Differences."""
    return float(np.sum(np.abs(pred - gt)))


def compute_mse(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((pred - gt) ** 2))


def compute_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean Absolute Error (raw, not ×1000)."""
    return float(np.mean(np.abs(pred - gt)))


def compute_boundary_sad(pred: np.ndarray, gt: np.ndarray, width: int = BOUNDARY_WIDTH) -> float:
    """
    Compute Boundary SAD: SAD restricted to pixels near the GT alpha edge.
    Boundary region = dilation(edge) ∩ erosion(edge) of GT alpha.
    Uses iterative binary dilation/erosion for simplicity.
    """
    from scipy.ndimage import binary_dilation, binary_erosion

    gt_binary = (gt > 0.5).astype(np.uint8)

    # Edge = dilated - eroded
    dilated = binary_dilation(gt_binary, iterations=width)
    eroded = binary_erosion(gt_binary, iterations=width)
    boundary_mask = dilated.astype(bool) & ~eroded.astype(bool)

    if not np.any(boundary_mask):
        return 0.0

    return float(np.sum(np.abs(pred[boundary_mask] - gt[boundary_mask])))


def compute_all_metrics(
    manifest_path: Path,
    predictions_dir: Path,
    methods: list[str],
) -> tuple[list[dict], list[dict]]:
    """
    Compute metrics for all methods and samples.

    Returns:
        (metrics_rows, leaderboard_rows)
    """
    # Read manifest
    with open(manifest_path, newline="", encoding="utf-8") as f:
        samples = list(csv.DictReader(f))

    metrics_rows = []
    leaderboard_rows = []

    for method_name in methods:
        print(f"\n{'='*60}")
        print(f"Computing metrics for method: {method_name}")
        print(f"{'='*60}")

        method_sad_list = []
        method_mse_list = []
        method_mae_list = []
        method_boundary_sad_list = []
        n_ok = 0
        n_fail = 0

        for sample in tqdm(samples, desc=f"[{method_name}] Metrics"):
            sample_id = sample["sample_id"]

            # Predicted alpha
            pred_path = predictions_dir / method_name / sample_id / "alpha.png"

            # GT alpha
            gt_path = fix_path(sample["alpha_path"])

            status = "ok"
            sad = float("nan")
            mse = float("nan")
            mae_x1000 = float("nan")
            boundary_sad = float("nan")

            try:
                if not pred_path.exists():
                    raise FileNotFoundError(f"Prediction missing: {pred_path}")

                if not gt_path.exists():
                    raise FileNotFoundError(f"GT missing: {gt_path}")

                pred_alpha = load_alpha(pred_path)
                gt_alpha = load_alpha(gt_path)

                # Resize pred to match GT if needed
                if pred_alpha.shape != gt_alpha.shape:
                    pred_img = Image.fromarray((pred_alpha * 255).astype(np.uint8))
                    pred_img = pred_img.resize(
                        (gt_alpha.shape[1], gt_alpha.shape[0]),
                        Image.LANCZOS,
                    )
                    pred_alpha = np.array(pred_img).astype(np.float32) / 255.0

                sad = compute_sad(pred_alpha, gt_alpha)
                mse = compute_mse(pred_alpha, gt_alpha)
                mae = compute_mae(pred_alpha, gt_alpha)
                mae_x1000 = mae * 1000.0
                boundary_sad = compute_boundary_sad(pred_alpha, gt_alpha)

                method_sad_list.append(sad)
                method_mse_list.append(mse)
                method_mae_list.append(mae_x1000)
                method_boundary_sad_list.append(boundary_sad)
                n_ok += 1

            except Exception as e:
                print(f"\n[WARN] sample_id={sample_id}: {e}")
                status = "failure"
                n_fail += 1

            metrics_rows.append(
                {
                    "sample_id": sample_id,
                    "method": method_name,
                    "status": status,
                    "SAD": f"{sad:.6f}" if not np.isnan(sad) else "",
                    "MSE": f"{mse:.12f}" if not np.isnan(mse) else "",
                    "MAE_x1000": f"{mae_x1000:.6f}" if not np.isnan(mae_x1000) else "",
                    "Boundary_SAD": f"{boundary_sad:.6f}" if not np.isnan(boundary_sad) else "",
                    "alpha_path_pred": str(pred_path),
                }
            )

        # Compute aggregate metrics
        total = n_ok + n_fail
        failure_rate = n_fail / total if total > 0 else 0.0

        if n_ok > 0:
            avg_sad = np.mean(method_sad_list)
            avg_mse = np.mean(method_mse_list)
            avg_mae = np.mean(method_mae_list)
            avg_boundary_sad = np.mean(method_boundary_sad_list)
        else:
            avg_sad = float("nan")
            avg_mse = float("nan")
            avg_mae = float("nan")
            avg_boundary_sad = float("nan")

        leaderboard_rows.append(
            {
                "method": method_name,
                "n": total,
                "ok": n_ok,
                "failure_rate": f"{failure_rate:.6f}",
                "SAD": f"{avg_sad:.6f}" if not np.isnan(avg_sad) else "",
                "MSE": f"{avg_mse:.12f}" if not np.isnan(avg_mse) else "",
                "MAE_x1000": f"{avg_mae:.6f}" if not np.isnan(avg_mae) else "",
                "Boundary_SAD": f"{avg_boundary_sad:.6f}" if not np.isnan(avg_boundary_sad) else "",
            }
        )

        print(f"\n[{method_name}] n={total}, ok={n_ok}, failure_rate={failure_rate:.4f}")
        print(f"  SAD={avg_sad:.6f}, MSE={avg_mse:.12f}, MAE_x1000={avg_mae:.6f}, Boundary_SAD={avg_boundary_sad:.6f}")

    return metrics_rows, leaderboard_rows


def write_csv(rows: list[dict], path: Path, fieldnames: list[str]):
    """Write rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Saved {len(rows)} rows → {path}")


def main():
    ap = argparse.ArgumentParser(description="Compute SAM2 benchmark metrics")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_VALID706,
        help="Path to manifest CSV",
    )
    ap.add_argument(
        "--predictions-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Root directory containing per-method alpha outputs",
    )
    ap.add_argument(
        "--methods",
        type=str,
        default="sam2_bbox_binary,sam2_guided",
        help="Comma-separated method names",
    )
    ap.add_argument(
        "--output-metrics-all",
        type=Path,
        default=ALIGNMENT_PACKAGE / "metrics_all.csv",
        help="Output path for metrics_all.csv",
    )
    ap.add_argument(
        "--output-leaderboard",
        type=Path,
        default=ALIGNMENT_PACKAGE / "leaderboard.csv",
        help="Output path for leaderboard.csv",
    )
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    if not args.manifest.exists():
        print(f"ERROR: Manifest not found: {args.manifest}")
        sys.exit(1)

    if not args.predictions_dir.exists():
        print(f"ERROR: Predictions directory not found: {args.predictions_dir}")
        sys.exit(1)

    print(f"Manifest: {args.manifest}")
    print(f"Predictions: {args.predictions_dir}")
    print(f"Methods: {methods}")

    metrics_rows, leaderboard_rows = compute_all_metrics(
        args.manifest,
        args.predictions_dir,
        methods,
    )

    # Fieldnames matching the required schemas
    metrics_fieldnames = [
        "sample_id",
        "method",
        "status",
        "SAD",
        "MSE",
        "MAE_x1000",
        "Boundary_SAD",
        "alpha_path_pred",
    ]
    leaderboard_fieldnames = [
        "method",
        "n",
        "ok",
        "failure_rate",
        "SAD",
        "MSE",
        "MAE_x1000",
        "Boundary_SAD",
    ]

    write_csv(metrics_rows, args.output_metrics_all, metrics_fieldnames)
    write_csv(leaderboard_rows, args.output_leaderboard, leaderboard_fieldnames)

    # Quick validation
    print(f"\n{'='*60}")
    print("Quick validation:")
    for lb in leaderboard_rows:
        ok_count = lb["ok"]
        expected = EXPECTED_N_VALID706
        if ok_count != expected:
            print(f"  ⚠ {lb['method']}: ok={ok_count} (expected {expected}) — MISSING SAMPLES!")
        else:
            print(f"  ✓ {lb['method']}: ok={ok_count} (matches expected {expected})")

    print("\n[SUCCESS] Metrics computation complete.")


if __name__ == "__main__":
    main()
