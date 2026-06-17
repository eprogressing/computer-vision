#!/usr/bin/env python3
"""
SAM2 Inference Script
=====================
Runs SAM2 with bbox prompt on all samples in the manifest,
producing alpha masks at outputs/{method}/{sample_id}/alpha.png

Two method variants:
  - sam2_bbox_binary: SAM2 mask → binary threshold → alpha
  - sam2_guided:      SAM2 mask → guided filter refinement → alpha

Usage:
  python sam2_inference.py [--manifest PATH] [--method METHOD_NAME]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm

# Add parent dir so we can import config
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    MANIFEST_VALID706,
    OUTPUT_DIR,
    SAM2_MODEL_CFG,
    SAM2_CHECKPOINT,
    METHODS,
    DEVICE,
    DATASET_BASE_OVERRIDE,
)


def fix_image_path(manifest_path_str: str) -> Path:
    """Convert manifest Linux path to local path if override is set."""
    p = Path(manifest_path_str)
    if DATASET_BASE_OVERRIDE is not None:
        # Extract the part after 'MicroMat3K_hf/'
        parts = p.parts
        try:
            idx = parts.index("MicroMat3K_hf")
            relative = Path(*parts[idx + 1 :])
            return DATASET_BASE_OVERRIDE / relative
        except ValueError:
            pass
    return p


def parse_bbox(bbox_str: str) -> np.ndarray:
    """Parse bbox string like '[x1, y1, x2, y2]' or '[x1 y1 x2 y2]' to xyxy array."""
    s = bbox_str.strip("[]")
    parts = [float(x.strip()) for x in s.replace(",", " ").split()]
    return np.array(parts)  # [x1, y1, x2, y2]


def parse_points(points_str: str) -> np.ndarray:
    """Parse points string like '[[x1,y1],[x2,y2]]' or '[]' to (N,2) array."""
    s = points_str.strip()
    if s == "[]" or s == "":
        return np.empty((0, 2), dtype=np.float32)
    # Remove outer brackets and split by '],['
    s = s.strip("[]")
    parts = s.split("],[")
    pts = []
    for p in parts:
        xy = [float(x.strip()) for x in p.replace("[", "").replace("]", "").split(",")]
        pts.append(xy)
    return np.array(pts, dtype=np.float32)


def rerank_masks(candidates: list[dict], bbox_xyxy: np.ndarray) -> np.ndarray:
    """Pick best mask: SAM2 IoU score (primary) + structural priors (secondary)."""
    x1, y1, x2, y2 = bbox_xyxy.astype(int)
    x1, x2 = max(0, x1), min(99999, x2)
    y1, y2 = max(0, y1), min(99999, y2)
    bbox_area = float(max(1, (x2 - x1) * (y2 - y1)))

    def score(m):
        binary = (m["mask"] > 0.5).astype(np.uint8)
        # Primary: SAM2's own trained IoU score (normalized)
        sam2_iou = m.get("iou_score", 0.9)
        # Secondary: prefer single connected region
        n_labels = cv2.connectedComponents(binary)[0]
        connectivity = 1.0 / max(n_labels - 1, 1)
        # Secondary: mask should fill reasonable portion of bbox (10-70%)
        mask_ratio = binary.sum() / bbox_area
        size_ok = 1.0 - abs(mask_ratio - 0.25) * 2.0
        size_ok = max(0.0, min(1.0, size_ok))
        return sam2_iou * 0.55 + connectivity * 0.25 + size_ok * 0.20

    scores = [score(c) for c in candidates]
    best_idx = int(np.argmax(scores))
    return candidates[best_idx]["mask"]


def predict_ensemble_rerank(
    predictor, image_rgb: np.ndarray, bbox_xyxy: np.ndarray,
    pos_pts: np.ndarray, neg_pts: np.ndarray,
) -> np.ndarray:
    """
    Prompt Ensemble + Mask Rerank (robust fallback version).
    Always succeeds: falls back to bbox-only on any error.
    """
    candidates = []

    # Always-safe: bbox-only (3 candidates)
    predictor.set_image(image_rgb)
    m, s, _ = predictor.predict(box=bbox_xyxy[None, :], multimask_output=True)
    for i in range(len(m)):
        candidates.append({"mask": m[i].astype(np.float32), "source": "bbox", "iou_score": float(s[i])})

    # Best-effort: bbox + positive points
    if len(pos_pts) > 0:
        try:
            predictor.set_image(image_rgb)
            m2, s2, _ = predictor.predict(
                box=bbox_xyxy[None, :],
                point_coords=pos_pts,
                point_labels=np.ones(len(pos_pts), dtype=np.int32),
                multimask_output=True,
            )
            for i in range(len(m2)):
                candidates.append({"mask": m2[i].astype(np.float32), "source": "bbox+pos", "iou_score": float(s2[i])})
        except Exception:
            pass

    # Rerank using SAM2 IoU score (primary) + structural priors
    best_mask = rerank_masks(candidates, bbox_xyxy)
    return best_mask


def build_sam2_predictor():
    """Build and return a SAM2ImagePredictor."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if not Path(SAM2_CHECKPOINT).exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found: {SAM2_CHECKPOINT}\n"
            f"Download from: https://github.com/facebookresearch/sam2#model-checkpoints"
        )
    # SAM2_MODEL_CFG is a Hydra config name (e.g. "sam2.1_hiera_b+.yaml"),
    # resolved via pkg://sam2 config path, not a filesystem path.

    print(f"[INFO] Building SAM2 predictor: {SAM2_MODEL_CFG}")
    sam2_model = build_sam2(SAM2_MODEL_CFG, SAM2_CHECKPOINT, device=DEVICE)
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor


def load_image_as_rgb(image_path: Path) -> np.ndarray:
    """Load image as RGB numpy array."""
    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def guided_filter_refine(image: np.ndarray, mask: np.ndarray, r: int = 1, eps: float = 1e-6) -> np.ndarray:
    """
    Apply guided filter to refine the SAM2 mask using the original image as guidance.
    This is a simplified version matching the PromptMatte-TTA-GF approach.

    Args:
        image: RGB image (H, W, 3) in [0, 255]
        mask:  Predicted mask (H, W) in [0, 1]
        r:     Filter radius (r=1 matches the GF r1 in the paper)
        eps:   Regularization

    Returns:
        Refined alpha (H, W) in [0, 1]
    """
    # Use OpenCV's guided filter if available, otherwise scipy-based fallback
    try:
        import cv2

        guide = image.astype(np.float32)
        src = mask.astype(np.float32)
        refined = cv2.ximgproc.guidedFilter(guide, src, r, eps)
        return np.clip(refined, 0, 1)
    except (ImportError, AttributeError):
        # Fallback: box-filter based guided filter
        from scipy.ndimage import uniform_filter

        guide = image.astype(np.float32) / 255.0
        src = mask.astype(np.float32)

        if len(guide.shape) == 3:
            guide_gray = np.mean(guide, axis=2)
        else:
            guide_gray = guide

        # Mean of guide, src, guide*src, guide*guide
        ksize = 2 * r + 1
        mean_guide = uniform_filter(guide_gray, ksize)
        mean_src = uniform_filter(src, ksize)
        mean_guide_src = uniform_filter(guide_gray * src, ksize)
        mean_guide_sq = uniform_filter(guide_gray * guide_gray, ksize)

        cov_guide_src = mean_guide_src - mean_guide * mean_src
        var_guide = mean_guide_sq - mean_guide * mean_guide

        a = cov_guide_src / (var_guide + eps)
        b = mean_src - a * mean_guide

        mean_a = uniform_filter(a, ksize)
        mean_b = uniform_filter(b, ksize)

        refined = mean_a * guide_gray + mean_b
        return np.clip(refined, 0, 1)


def predict_multiscale(predictor, image_rgb, bbox_xyxy, scales=[0.75, 1.0, 1.25]):
    """
    Multi-scale fusion: run SAM2 at multiple scales and average the masks.
    """
    h, w = image_rgb.shape[:2]
    all_masks = []

    for scale in scales:
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image_rgb, (new_w, new_h))
        bbox_scaled = bbox_xyxy * scale

        predictor.set_image(resized)
        masks, scores, _ = predictor.predict(
            point_coords=None, point_labels=None,
            box=bbox_scaled[None, :], multimask_output=True,
        )
        best = masks[np.argmax(scores)]
        # Resize back to original resolution
        restored = cv2.resize(best.astype(np.float32), (w, h))
        all_masks.append(restored)

    fused = np.mean(all_masks, axis=0)
    return fused


def predict_ensemble_multiscale(
    predictor, image_rgb: np.ndarray, bbox_xyxy: np.ndarray,
    pos_pts: np.ndarray, neg_pts: np.ndarray,
    scales: list[float] = [0.75, 1.0, 1.25],
) -> np.ndarray:
    """
    Ensemble rerank at multiple scales and average the results.
    Combines the robustness of ensemble rerank with multi-scale stability.
    """
    h, w = image_rgb.shape[:2]
    all_masks = []

    for scale in scales:
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image_rgb, (new_w, new_h))
        bbox_scaled = bbox_xyxy * scale

        # Run ensemble rerank on this scale
        mask = predict_ensemble_rerank(predictor, resized, bbox_scaled, pos_pts, neg_pts)

        # Resize back to original resolution
        restored = cv2.resize(mask, (w, h))
        all_masks.append(restored)

    fused = np.mean(all_masks, axis=0)
    return fused


def run_inference(
    predictor,
    manifest_path: Path,
    method_name: str,
    binary_threshold: float | None,
    multiscale_scales: list[float] | None = None,
    ensemble_rerank: bool = False,
) -> dict:
    """
    Run SAM2 inference for one method on all samples.

    Returns:
        dict mapping sample_id -> {"alpha_path": str, "status": "ok"|"failure"}
    """
    # Read manifest
    with open(manifest_path, newline="", encoding="utf-8") as f:
        samples = list(csv.DictReader(f))

    output_dir = OUTPUT_DIR / method_name
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    n_ok = 0
    n_fail = 0

    for sample in tqdm(samples, desc=f"[{method_name}] Inference"):
        sample_id = sample["sample_id"]
        image_path = fix_image_path(sample["image_path"])
        bbox_str = sample["bbox"]

        alpha_output_path = output_dir / sample_id / "alpha.png"
        alpha_output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # --- Load image ---
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            image_rgb = load_image_as_rgb(image_path)

            # --- Parse bbox & points ---
            bbox_xyxy = parse_bbox(bbox_str)  # [x1, y1, x2, y2]
            pos_pts = parse_points(sample.get("positive_points", "[]"))
            neg_pts = parse_points(sample.get("negative_points", "[]"))

            # --- SAM2 predict ---
            if ensemble_rerank and multiscale_scales:
                mask = predict_ensemble_multiscale(predictor, image_rgb, bbox_xyxy, pos_pts, neg_pts, scales=multiscale_scales)
            elif ensemble_rerank:
                mask = predict_ensemble_rerank(predictor, image_rgb, bbox_xyxy, pos_pts, neg_pts)
            elif multiscale_scales:
                predictor.set_image(image_rgb)
                mask = predict_multiscale(predictor, image_rgb, bbox_xyxy, scales=multiscale_scales)
            else:
                predictor.set_image(image_rgb)
                masks, scores, logits = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=bbox_xyxy[None, :],
                    multimask_output=True,
                )
                best_idx = np.argmax(scores)
                mask = masks[best_idx].astype(np.float32)

            # --- Produce alpha ---
            if binary_threshold is not None:
                # Binary alpha: already 0/1 from SAM2
                alpha = mask
            else:
                # Soft alpha via guided filter refinement
                alpha = guided_filter_refine(image_rgb, mask, r=1)

            # --- Save alpha.png ---
            alpha_uint8 = (alpha * 255).astype(np.uint8)
            Image.fromarray(alpha_uint8, mode="L").save(alpha_output_path)

            results[sample_id] = {
                "alpha_path": str(alpha_output_path),
                "status": "ok",
            }
            n_ok += 1

        except Exception as e:
            print(f"\n[ERROR] sample_id={sample_id}: {e}")
            # Save a zero alpha as fallback
            h, w = 1024, 1024  # fallback size
            try:
                img = load_image_as_rgb(image_path)
                h, w = img.shape[:2]
            except Exception:
                pass
            alpha_output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.zeros((h, w), dtype=np.uint8), mode="L").save(alpha_output_path)

            results[sample_id] = {
                "alpha_path": str(alpha_output_path),
                "status": "failure",
            }
            n_fail += 1

    print(f"\n[{method_name}] Done: {n_ok} ok, {n_fail} failed, {len(samples)} total")
    return results


def main():
    ap = argparse.ArgumentParser(description="SAM2 Inference Pipeline")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_VALID706,
        help="Path to manifest CSV",
    )
    ap.add_argument(
        "--method",
        type=str,
        default=None,
        help="Run only this method (default: all methods in config)",
    )
    ap.add_argument(
        "--save-results",
        type=Path,
        default=None,
        help="Save per-sample results JSON (for later metrics computation)",
    )
    args = ap.parse_args()

    # Build SAM2 predictor once
    predictor = build_sam2_predictor()

    # Select methods
    methods_to_run = METHODS
    if args.method:
        methods_to_run = [m for m in METHODS if m["name"] == args.method]
        if not methods_to_run:
            print(f"ERROR: Method '{args.method}' not found in config.METHODS")
            sys.exit(1)

    all_results = {}

    for method_cfg in methods_to_run:
        name = method_cfg["name"]
        threshold = method_cfg.get("binary_threshold")
        scales = method_cfg.get("multiscale", None)
        use_ensemble = method_cfg.get("ensemble_rerank", False)
        results = run_inference(
            predictor, args.manifest, name, threshold,
            multiscale_scales=scales, ensemble_rerank=use_ensemble,
        )
        all_results[name] = results

    # Save results mapping
    if args.save_results:
        import json as _json

        args.save_results.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_results, "w") as f:
            _json.dump(all_results, f, indent=2)
        print(f"[INFO] Results saved to {args.save_results}")

    print("\n[SUCCESS] SAM2 inference complete.")


if __name__ == "__main__":
    main()
