#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


ROOT = Path("/home/lpy/anisorisk/computer_vison")
FINAL_PACKAGE = ROOT / "runs/final_submission_package_20260604_161922"
ZIM_RUN = ROOT / "runs/protocol_repair_final_claim_20260602_174358"
BSP_RUN = ROOT / "runs/box_prior_fast_eval_20260604_092617"
COMPOSER_RUN = ROOT / "runs/promptmatte_composer_showcase_20260604"

BASELINE = "zim_vitb_bbox_default"
FLIP_TTA = "zim_vitb_flip_tta_bbox"
GUIDED = "zim_vitb_bbox_default_guided_r1"
TTA_GF = "zim_vitb_flip_tta_bbox_guided_r1"
BSP = "promptmatte_tta_gf_box_support_prior"

METHOD_LABELS = {
    BASELINE: "ZIM bbox baseline",
    FLIP_TTA: "Flip TTA",
    GUIDED: "Guided filter",
    TTA_GF: "PromptMatte-TTA-GF",
    BSP: "TTA-GF+BSP",
}
METRICS = ["SAD", "MSE", "MAE_x1000", "Boundary_SAD"]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def fmt(value: Any) -> str:
    v = safe_float(value)
    if math.isnan(v):
        return str(value) if value not in (None, "") else ""
    if abs(v) < 0.001 and v != 0:
        return f"{v:.9f}"
    return f"{v:.6f}"


def md_table(rows: list[dict[str, Any]], keys: list[str], limit: int | None = None) -> str:
    if not rows:
        return "_No rows._\n"
    use_rows = rows if limit is None else rows[:limit]
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in use_rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |")
    return "\n".join(lines) + "\n"


def copy_file(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_dir(src: Path, dst: Path, ignore: shutil.IgnorePattern | None = None) -> bool:
    if not src.exists():
        return False
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore)
    return True


def alpha_array(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    img = Image.open(path).convert("L")
    if size and img.size != size:
        img = img.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def alpha_image(path: Path, size: tuple[int, int]) -> Image.Image:
    arr = alpha_array(path, size)
    return Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8), "L").convert("RGB")


def error_heatmap(pred_path: Path, gt_path: Path, size: tuple[int, int]) -> Image.Image:
    pred = alpha_array(pred_path, size)
    gt = alpha_array(gt_path, size)
    err = np.abs(pred - gt)
    vmax = max(0.08, float(np.percentile(err, 99)))
    norm = np.clip(err / vmax, 0, 1)
    cmap = plt.get_cmap("magma")
    rgba = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGB")


def boundary_overlay(rgb_path: Path, pred_path: Path, gt_path: Path, size: tuple[int, int]) -> Image.Image:
    rgb = Image.open(rgb_path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    pred = alpha_array(pred_path, size)
    gt = alpha_array(gt_path, size)
    p = pred >= 0.5
    g = gt >= 0.5
    edge_p = np.zeros_like(p, dtype=bool)
    edge_g = np.zeros_like(g, dtype=bool)
    edge_p[1:, :] |= p[1:, :] != p[:-1, :]
    edge_p[:-1, :] |= p[:-1, :] != p[1:, :]
    edge_p[:, 1:] |= p[:, 1:] != p[:, :-1]
    edge_p[:, :-1] |= p[:, :-1] != p[:, 1:]
    edge_g[1:, :] |= g[1:, :] != g[:-1, :]
    edge_g[:-1, :] |= g[:-1, :] != g[1:, :]
    edge_g[:, 1:] |= g[:, 1:] != g[:, :-1]
    edge_g[:, :-1] |= g[:, :-1] != g[:, 1:]
    arr = np.asarray(rgb).copy()
    arr[edge_g] = [0, 220, 0]
    arr[edge_p] = [255, 40, 40]
    both = edge_g & edge_p
    arr[both] = [255, 220, 0]
    return Image.fromarray(arr, "RGB")


def parse_bbox(row: dict[str, str]) -> list[int]:
    return [int(round(float(x))) for x in json.loads(row["bbox"])]


def expand_bbox(bbox: list[int], width: int, height: int, pad: float) -> list[int]:
    x1, y1, x2, y2 = [float(x) for x in bbox]
    bw = max(1.0, x2 - x1 + 1.0)
    bh = max(1.0, y2 - y1 + 1.0)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return [
        int(max(0, math.floor(cx - bw * pad / 2.0))),
        int(max(0, math.floor(cy - bh * pad / 2.0))),
        int(min(width - 1, math.ceil(cx + bw * pad / 2.0))),
        int(min(height - 1, math.ceil(cy + bh * pad / 2.0))),
    ]


def support_mask(size: tuple[int, int], bbox: list[int], pad: float = 1.5, blur: float = 8.0) -> np.ndarray:
    width, height = size
    x1, y1, x2, y2 = expand_bbox(bbox, width, height, pad)
    arr = np.zeros((height, width), dtype=np.uint8)
    arr[y1 : y2 + 1, x1 : x2 + 1] = 255
    img = Image.fromarray(arr, "L").filter(ImageFilter.GaussianBlur(radius=blur))
    return np.asarray(img, dtype=np.float32) / 255.0


def apply_bsp_alpha(alpha: np.ndarray, row: dict[str, str]) -> np.ndarray:
    mask = support_mask((alpha.shape[1], alpha.shape[0]), parse_bbox(row), pad=1.5, blur=8.0)
    gate = mask + (1.0 - mask) * 0.2
    out = alpha * (1.0 - 0.10 + 0.10 * gate)
    return np.clip(out, 0, 1)


def save_alpha(path: Path, alpha: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(alpha, 0, 1) * 255 + 0.5).astype(np.uint8), "L").save(path)


def thumbnail(img: Image.Image, size: tuple[int, int] = (260, 260)) -> Image.Image:
    return ImageOps.contain(img.convert("RGB"), size, Image.Resampling.LANCZOS)


def captioned(img: Image.Image, title: str, subtitle: str = "", width: int = 300, height: int = 350) -> Image.Image:
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
        font_sub = ImageFont.truetype("DejaVuSans.ttf", 13)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
    thumb = thumbnail(img, (width - 20, height - 78))
    x = (width - thumb.width) // 2
    y = 42 + (height - 78 - thumb.height) // 2
    canvas.paste(thumb, (x, y))
    draw.text((10, 10), title[:30], fill=(20, 30, 45), font=font_title)
    if subtitle:
        draw.text((10, height - 28), subtitle[:45], fill=(70, 80, 95), font=font_sub)
    return canvas


def make_grid(cells: list[Image.Image], cols: int, out: Path, gap: int = 10, bg: str = "#F8FAFC") -> None:
    if not cells:
        return
    rows = math.ceil(len(cells) / cols)
    w, h = cells[0].size
    canvas = Image.new("RGB", (cols * w + (cols + 1) * gap, rows * h + (rows + 1) * gap), bg)
    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        canvas.paste(cell, (gap + c * (w + gap), gap + r * (h + gap)))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=95)


def make_table_image(path: Path, rows: list[dict[str, Any]], cols: list[str], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig_w = max(10, len(cols) * 1.55)
    fig_h = max(3.0, len(rows) * 0.48 + 1.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220)
    ax.axis("off")
    ax.set_title(title, fontsize=16, weight="bold", pad=14)
    cells = []
    for row in rows:
        cells.append([fmt(row.get(c, "")) if c in METRICS + ["failure_rate", "relative_improvement_percent"] else str(row.get(c, "")) for c in cols])
    table = ax.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.4)
    for (r, _c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#E0F2FE")
            cell.set_text_props(weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F8FAFC")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_bar_chart(path: Path, leaderboard: list[dict[str, str]]) -> None:
    methods = [r["display_method"] for r in leaderboard]
    sad = [safe_float(r["SAD"]) for r in leaderboard]
    bnd = [safe_float(r["Boundary_SAD"]) for r in leaderboard]
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), dpi=220)
    for ax, vals, title in [(axes[0], sad, "SAD lower is better"), (axes[1], bnd, "Boundary SAD lower is better")]:
        bars = ax.bar(x, vals, color=["#94A3B8", "#60A5FA", "#A78BFA", "#34D399", "#F59E0B"])
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=8)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.grid(axis="y", alpha=0.25)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Final valid706 leaderboard", fontsize=15, weight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def load_maps() -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    metrics = read_csv(ZIM_RUN / "metrics/final_metrics_all.csv")
    metric_map = {(r["sample_id"], r["method"]): r for r in metrics}
    manifest = {r["sample_id"]: r for r in read_csv(BSP_RUN / "manifests/all_valid706.csv")}
    bsp = read_csv(BSP_RUN / "metrics/all_valid706_box_prior_metrics.csv")
    bsp_map = {(r["sample_id"], r["method"]): r for r in bsp}
    return metric_map, manifest, bsp_map


def choose_examples() -> dict[str, list[str]]:
    improvements = read_csv(ZIM_RUN / "failure_analysis/top_improvements.csv")
    regressions = read_csv(ZIM_RUN / "failure_analysis/top_regressions.csv")
    bsp_rows = read_csv(BSP_RUN / "metrics/all_valid706_box_prior_metrics.csv")
    by_sid: dict[str, dict[str, dict[str, str]]] = {}
    for row in bsp_rows:
        by_sid.setdefault(row["sample_id"], {})[row["method"]] = row
    bsp_gains = []
    for sid, rows in by_sid.items():
        if TTA_GF in rows and BSP in rows:
            cur = safe_float(rows[TTA_GF].get("SAD"))
            new = safe_float(rows[BSP].get("SAD"))
            bcur = safe_float(rows[TTA_GF].get("Boundary_SAD"))
            bnew = safe_float(rows[BSP].get("Boundary_SAD"))
            gain = cur - new
            bgain = bcur - bnew
            if gain > 1e-6 or bgain > 1e-6:
                bsp_gains.append((gain + 0.5 * bgain, sid))
    bsp_gains.sort(reverse=True)
    metrics = read_csv(ZIM_RUN / "metrics/final_metrics_all.csv")
    by_sid_method = {(r["sample_id"], r["method"]): r for r in metrics}
    smoothing_gains = []
    for sid in {r["sample_id"] for r in metrics}:
        raw = by_sid_method.get((sid, FLIP_TTA))
        smooth = by_sid_method.get((sid, TTA_GF))
        if raw and smooth:
            gain = safe_float(raw.get("SAD")) - safe_float(smooth.get("SAD"))
            bgain = safe_float(raw.get("Boundary_SAD")) - safe_float(smooth.get("Boundary_SAD"))
            if gain > 1e-6 or bgain > 1e-6:
                smoothing_gains.append((gain + 0.75 * bgain, sid))
    smoothing_gains.sort(reverse=True)
    return {
        "baseline_vs_ttagf": [r["sample_id"] for r in improvements[:6]],
        "smoothing_effect": [sid for _, sid in smoothing_gains[:6]],
        "failure_cases": [r["sample_id"] for r in regressions[:4]],
        "ttagf_vs_bsp": [sid for _, sid in bsp_gains[:6]],
    }


def make_case_sheet(out_dir: Path, sid: str, methods: list[str], title: str) -> None:
    metric_map, manifest, _bsp_map = load_maps()
    row = manifest.get(sid)
    if not row:
        return
    image_path = Path(row["image_path"])
    gt_path = Path(row["alpha_path"])
    rgb = Image.open(image_path).convert("RGB")
    size = rgb.size
    cells = [
        captioned(rgb, "Input image", sid),
        captioned(alpha_image(gt_path, size), "GT alpha", "evaluation only"),
    ]
    for method in methods:
        metric = metric_map.get((sid, method))
        if not metric:
            continue
        pred_path = Path(metric["alpha_path_pred"])
        if not pred_path.exists():
            continue
        subtitle = f"SAD {safe_float(metric.get('SAD')):.3f} | B {safe_float(metric.get('Boundary_SAD')):.3f}"
        cells.append(captioned(alpha_image(pred_path, size), METHOD_LABELS[method], subtitle))
        cells.append(captioned(error_heatmap(pred_path, gt_path, size), "Error heatmap", METHOD_LABELS[method]))
    make_grid(cells, cols=4, out=out_dir / f"{sid}_{title}.jpg")


def make_bsp_sheet(out_dir: Path, sid: str) -> None:
    metric_map, manifest, bsp_map = load_maps()
    row = manifest.get(sid)
    if not row:
        return
    image_path = Path(row["image_path"])
    gt_path = Path(row["alpha_path"])
    rgb = Image.open(image_path).convert("RGB")
    size = rgb.size
    cur_metric = metric_map.get((sid, TTA_GF))
    bsp_metric = bsp_map.get((sid, BSP))
    if not cur_metric or not bsp_metric:
        return
    cur_path = Path(cur_metric["alpha_path_pred"])
    bsp_alpha = apply_bsp_alpha(alpha_array(cur_path, size), row)
    bsp_path = out_dir / "_generated_bsp_alpha" / f"{sid}_bsp_alpha.png"
    save_alpha(bsp_path, bsp_alpha)
    cells = [
        captioned(rgb, "Input image", sid),
        captioned(alpha_image(gt_path, size), "GT alpha", "evaluation only"),
        captioned(alpha_image(cur_path, size), "PromptMatte-TTA-GF", f"SAD {safe_float(cur_metric.get('SAD')):.3f}"),
        captioned(alpha_image(bsp_path, size), "TTA-GF+BSP", f"SAD {safe_float(bsp_metric.get('SAD')):.3f}"),
        captioned(error_heatmap(cur_path, gt_path, size), "TTA-GF error", "before BSP"),
        captioned(error_heatmap(bsp_path, gt_path, size), "BSP error", "small gain"),
        captioned(boundary_overlay(image_path, cur_path, gt_path, size), "Boundary overlay", "green GT / red pred"),
        captioned(rgb, "BSP effect", f"delta SAD {safe_float(cur_metric.get('SAD')) - safe_float(bsp_metric.get('SAD')):.4f}"),
    ]
    make_grid(cells, cols=4, out=out_dir / f"{sid}_ttagf_vs_bsp.jpg")


def generate_visuals(delivery: Path) -> None:
    examples = choose_examples()
    write_text(delivery / "visuals/selected_examples.json", json.dumps(examples, indent=2, ensure_ascii=False))

    base_dir = delivery / "visuals/baseline_vs_ttagf"
    smooth_dir = delivery / "visuals/smoothing_effect"
    fail_dir = delivery / "visuals/failure_cases"
    bsp_dir = delivery / "visuals/ttagf_vs_bsp"
    for sid in examples["baseline_vs_ttagf"]:
        make_case_sheet(base_dir, sid, [BASELINE, TTA_GF], "baseline_vs_ttagf")
    for sid in examples["smoothing_effect"]:
        make_case_sheet(smooth_dir, sid, [FLIP_TTA, TTA_GF], "smoothing_effect")
    for sid in examples["failure_cases"]:
        make_case_sheet(fail_dir, sid, [BASELINE, TTA_GF], "failure_case")
    for sid in examples["ttagf_vs_bsp"]:
        make_bsp_sheet(bsp_dir, sid)

    # Existing high-level contact sheets.
    copy_file(ZIM_RUN / "ppt_assets/top_improvements_contact_sheet.png", delivery / "visuals/baseline_vs_ttagf/top_improvements_contact_sheet.png")
    copy_file(ZIM_RUN / "ppt_assets/top_regressions_contact_sheet.png", delivery / "visuals/failure_cases/top_regressions_contact_sheet.png")

    composer_dst = delivery / "visuals/composer_showcase"
    composer_dst.mkdir(parents=True, exist_ok=True)
    for src in [COMPOSER_RUN / "visuals/overview_contact_sheet.jpg"]:
        copy_file(src, composer_dst / src.name)
    for sample_dir in sorted([p for p in (COMPOSER_RUN / "visuals").iterdir() if p.is_dir()]):
        out = composer_dst / sample_dir.name
        out.mkdir(parents=True, exist_ok=True)
        for name in ["original.jpg", "rgba.png", "replace_blue.jpg", "replace_white.jpg", "replace_gradient.jpg", "blur_bg.jpg", "alpha_edge.jpg", "contact_sheet.jpg"]:
            copy_file(sample_dir / name, out / name)


def generate_tables(delivery: Path) -> None:
    tables_dir = delivery / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    files = [
        FINAL_PACKAGE / "metrics/final_leaderboard_valid706.csv",
        FINAL_PACKAGE / "metrics/final_leaderboard_final_holdout506.csv",
        FINAL_PACKAGE / "metrics/final_method_improvements.csv",
        ZIM_RUN / "metrics/fair_ablation_table.csv",
        BSP_RUN / "metrics/all_valid706_box_prior_leaderboard.csv",
        BSP_RUN / "metrics/final_holdout506_box_prior_leaderboard.csv",
    ]
    for src in files:
        copy_file(src, tables_dir / src.name)
    leaderboard = read_csv(FINAL_PACKAGE / "metrics/final_leaderboard_valid706.csv")
    make_table_image(delivery / "ppt_assets/final_leaderboard_table.png", leaderboard, ["display_method", "n_ok", "SAD", "MSE", "MAE_x1000", "Boundary_SAD"], "Final valid706 leaderboard")
    make_bar_chart(delivery / "ppt_assets/final_leaderboard_bar_chart.png", leaderboard)
    improvements = read_csv(FINAL_PACKAGE / "metrics/final_method_improvements.csv")
    valid_imp = [r for r in improvements if r["split"] == "valid706" and r["comparison"] in ["PromptMatte-TTA-GF vs ZIM bbox baseline", "PromptMatte-TTA-GF+BSP vs ZIM bbox baseline", "BSP increment over PromptMatte-TTA-GF"]]
    make_table_image(delivery / "ppt_assets/improvement_table.png", valid_imp, ["comparison", "metric", "relative_improvement_percent"], "Relative improvements on valid706")
    copy_file(FINAL_PACKAGE / "ppt_assets/01_method_diagram.png", delivery / "ppt_assets/method_diagram.png")
    copy_file(FINAL_PACKAGE / "ppt_assets/03_ablation_table.png", delivery / "ppt_assets/ablation_table.png")
    copy_file(FINAL_PACKAGE / "ppt_assets/07_claim_safety.png", delivery / "ppt_assets/claim_safety.png")


def collect_code(delivery: Path) -> None:
    code_dir = delivery / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    wanted = [
        "scripts/final_claim",
        "scripts/box_prior",
        "scripts/final_submission",
        "scripts/final_delivery",
        "scripts/official_metrics.py",
    ]
    for rel in wanted:
        src = ROOT / rel
        dst = code_dir / rel
        if src.is_dir():
            copy_dir(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "._*"))
        elif src.exists():
            copy_file(src, dst)
    write_text(
        code_dir / "README_CODE.md",
        """# Code Included

This folder contains the scripts needed to reproduce the final packaging and the main post-processing logic:

- `scripts/final_claim`: final metric consolidation for ZIM baseline and PromptMatte-TTA-GF.
- `scripts/box_prior`: Box-Support Prior implementation and full evaluation script.
- `scripts/final_submission`: final leaderboard / claim-safety / SAM2 alignment package generation.
- `scripts/final_delivery`: final PPT/submission delivery packaging.
- `scripts/official_metrics.py`: local metric helper used by previous runs.

Large model weights, datasets, and generated alpha outputs are intentionally not copied into this folder.
""",
    )


def collect_reports(delivery: Path) -> None:
    reports = delivery / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    for src in [
        FINAL_PACKAGE / "reports/FINAL_SUBMISSION_SUMMARY.md",
        FINAL_PACKAGE / "claim_safety/FINAL_ALLOWED_CLAIMS.md",
        FINAL_PACKAGE / "claim_safety/FINAL_FORBIDDEN_CLAIMS.md",
        FINAL_PACKAGE / "reports/composer_showcase_summary.md",
        FINAL_PACKAGE / "sam2_integration/SAM2_RESULT_MISSING_TODO.md",
        ZIM_RUN / "FINAL_PROMPTMATTE_TTA_GF_SUMMARY.md",
        BSP_RUN / "FINAL_BOX_PRIOR_FAST_FULL_EVAL.md",
    ]:
        copy_file(src, reports / src.name)


def make_gallery(delivery: Path) -> None:
    entries = []
    for sub in ["baseline_vs_ttagf", "smoothing_effect", "ttagf_vs_bsp", "failure_cases", "composer_showcase"]:
        for img in sorted((delivery / "visuals" / sub).rglob("*.jpg")) + sorted((delivery / "visuals" / sub).rglob("*.png")):
            rel = img.relative_to(delivery)
            entries.append((sub, rel))
    body = ["<!doctype html><html><head><meta charset='utf-8'><title>PromptMatte Final Visual Gallery</title></head><body>"]
    body.append("<h1>PromptMatte Final Visual Gallery</h1>")
    body.append("<p>Benchmark visuals are separated from Composer application visuals.</p>")
    current = None
    for sub, rel in entries:
        if sub != current:
            body.append(f"<h2>{html.escape(sub)}</h2>")
            current = sub
        body.append(f"<figure><img src='{html.escape(str(rel))}' style='max-width:1100px;width:95%;border:1px solid #ddd'><figcaption>{html.escape(str(rel))}</figcaption></figure>")
    body.append("</body></html>")
    write_text(delivery / "visuals/index.html", "\n".join(body))


def make_readme(delivery: Path) -> None:
    leaderboard = read_csv(FINAL_PACKAGE / "metrics/final_leaderboard_valid706.csv")
    imp = read_csv(FINAL_PACKAGE / "metrics/final_method_improvements.csv")
    core_imp = [r for r in imp if r["split"] == "valid706" and r["comparison"] in ["PromptMatte-TTA-GF vs ZIM bbox baseline", "PromptMatte-TTA-GF+BSP vs ZIM bbox baseline", "BSP increment over PromptMatte-TTA-GF"]]
    text = f"""# Computer Vision Final Project: PromptMatte

This is the final PPT/submission package for the PromptMatte computer vision project.

## Final method

`PromptMatte-TTA-GF+BSP`

- `PromptMatte-TTA-GF`: ZIM ViT-B official bbox prompt + horizontal flip TTA + guided filter r1.
- `BSP`: Box-Support Prior, a no-GT spatial support prior derived from the official bbox prompt.
- `Composer`: application-only showcase for RGBA export, background replacement, blur background, and alpha-edge visualization.

## What to use for PPT

- Main table: `tables/final_leaderboard_valid706.csv`
- Main table image: `ppt_assets/final_leaderboard_table.png`
- Bar chart: `ppt_assets/final_leaderboard_bar_chart.png`
- Method diagram: `ppt_assets/method_diagram.png`
- Ablation table: `ppt_assets/ablation_table.png`
- Claim safety: `ppt_assets/claim_safety.png`
- ZIM vs PromptMatte visual cases: `visuals/baseline_vs_ttagf/`
- Flip TTA vs guided smoothing cases: `visuals/smoothing_effect/`
- PromptMatte-TTA-GF vs BSP cases: `visuals/ttagf_vs_bsp/`
- Failure cases: `visuals/failure_cases/`
- Composer showcase: `visuals/composer_showcase/`
- Visual gallery: `visuals/index.html`

## Final valid706 leaderboard

{md_table(leaderboard, ["display_method", "n_ok", "SAD", "MSE", "MAE_x1000", "Boundary_SAD"], None)}

## Improvement summary

{md_table(core_imp, ["comparison", "metric", "relative_improvement_percent"], None)}

## How to explain the visual results

1. ZIM baseline gives the starting alpha matte from official bbox prompt.
2. Horizontal flip TTA stabilizes prediction by averaging original and flipped inference.
3. Guided filtering improves local boundary consistency using the image as guidance.
4. BSP provides a conservative box-derived support prior, giving a small but consistent metric gain.
5. Composer shows the predicted alpha is usable as an editing primitive, not just a benchmark mask.

## Claim safety

- This is a valid706 official-prompt subset result, not full MicroMat-3K.
- Composer images are demos and are not benchmark metrics.
- Approximate Grad/Conn must not be reported as official full metrics.
- SAM2 teammate results are not included unless an aligned 706-row submission is provided.

## Reproduction

Remote source directory:

`/home/lpy/anisorisk/computer_vison`

Final source runs:

- `runs/protocol_repair_final_claim_20260602_174358`
- `runs/box_prior_fast_eval_20260604_092617`
- `runs/promptmatte_composer_showcase_20260604`
- `runs/final_submission_package_20260604_161922`

The package was generated by:

```bash
cd /home/lpy/anisorisk/computer_vison
python3 scripts/final_delivery/make_final_delivery.py
```
"""
    write_text(delivery / "README.md", text)


def verify_delivery(delivery: Path) -> None:
    checks = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    required = [
        "README.md",
        "tables/final_leaderboard_valid706.csv",
        "ppt_assets/final_leaderboard_table.png",
        "ppt_assets/final_leaderboard_bar_chart.png",
        "ppt_assets/method_diagram.png",
        "visuals/index.html",
        "reports/FINAL_SUBMISSION_SUMMARY.md",
        "code/README_CODE.md",
    ]
    for rel in required:
        p = delivery / rel
        add(rel, p.exists() and p.stat().st_size > 0, str(p.stat().st_size) if p.exists() else "missing")
    for sub in ["baseline_vs_ttagf", "smoothing_effect", "ttagf_vs_bsp", "failure_cases", "composer_showcase"]:
        count = len(list((delivery / "visuals" / sub).rglob("*.jpg"))) + len(list((delivery / "visuals" / sub).rglob("*.png")))
        add(f"visual_count_{sub}", count > 0, str(count))
    write_csv(delivery / "VERIFY_DELIVERY.csv", checks)
    failed = [c for c in checks if c["status"] != "PASS"]
    if failed:
        raise RuntimeError(f"delivery verification failed: {failed}")


def make_package(args: argparse.Namespace) -> Path:
    run_id = args.run_id or f"final_delivery_computer_vision_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = ROOT / "runs" / run_id
    delivery = run_dir / "computer_vision"
    delivery.mkdir(parents=True, exist_ok=False)
    for sub in ["code", "tables", "ppt_assets", "visuals", "reports", "repro", "logs"]:
        (delivery / sub).mkdir(parents=True, exist_ok=True)

    generate_tables(delivery)
    generate_visuals(delivery)
    collect_code(delivery)
    collect_reports(delivery)
    make_gallery(delivery)
    make_readme(delivery)
    copy_file(FINAL_PACKAGE / "repro/commands_reproduce_final_package.sh", delivery / "repro/commands_reproduce_final_package.sh")
    verify_delivery(delivery)

    write_text(run_dir / "DONE.txt", f"FINAL COMPUTER_VISION DELIVERY DONE at {time.strftime('%Y-%m-%d %H:%M:%S')}\nDELIVERY={delivery}\n")
    print((run_dir / "DONE.txt").read_text(encoding="utf-8").strip())
    return delivery


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()
    make_package(args)


if __name__ == "__main__":
    main()
