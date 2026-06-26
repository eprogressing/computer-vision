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
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/home/lpy/anisorisk/computer_vison")
ZIM_RUN = ROOT / "runs/protocol_repair_final_claim_20260602_174358"
BSP_RUN = ROOT / "runs/box_prior_fast_eval_20260604_092617"
COMPOSER_RUN = ROOT / "runs/promptmatte_composer_showcase_20260604"

BASELINE = "zim_vitb_bbox_default"
FLIP_TTA = "zim_vitb_flip_tta_bbox"
GUIDED = "zim_vitb_bbox_default_guided_r1"
TTA_GF = "zim_vitb_flip_tta_bbox_guided_r1"
BSP = "promptmatte_tta_gf_box_support_prior"

METHOD_ORDER = [BASELINE, FLIP_TTA, GUIDED, TTA_GF, BSP]
DISPLAY = {
    BASELINE: "ZIM bbox baseline",
    FLIP_TTA: "ZIM bbox + horizontal flip TTA",
    GUIDED: "ZIM bbox + guided filter r1",
    TTA_GF: "PromptMatte-TTA-GF",
    BSP: "PromptMatte-TTA-GF+BSP",
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
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


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
    out = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in use_rows:
        vals = []
        for key in keys:
            val = row.get(key, "")
            vals.append(fmt(val) if isinstance(val, (float, int)) or key in METRICS + ["failure_rate", "relative_improvement"] else str(val))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out) + "\n"


def mean_metric(rows: list[dict[str, str]], key: str) -> float:
    vals = [safe_float(r.get(key)) for r in rows if r.get("status", "ok") == "ok"]
    vals = [v for v in vals if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else math.nan


def ensure_dir_tree(run_dir: Path) -> None:
    for name in [
        "logs",
        "metrics",
        "reports",
        "visuals",
        "ppt_assets",
        "tables",
        "figures",
        "composer",
        "sam2_integration",
        "claim_safety",
        "repro",
    ]:
        (run_dir / name).mkdir(parents=True, exist_ok=True)


def clean_link_or_copy(src: Path, dst: Path, symlink: bool = True) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if symlink:
        dst.symlink_to(src)
    elif src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return True


def table_image(path: Path, rows: list[dict[str, Any]], cols: list[str], title: str, max_rows: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    shown = rows[:max_rows]
    cell_text = [[fmt(row.get(c, "")) if c in METRICS + ["failure_rate", "relative_improvement"] else str(row.get(c, "")) for c in cols] for row in shown]
    fig_w = max(9, len(cols) * 1.6)
    fig_h = max(2.8, 0.45 * len(shown) + 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=12)
    table = ax.table(cellText=cell_text, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    for (r, _c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#E8EEF7")
            cell.set_text_props(weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F8FAFC")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def method_diagram(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 3.2), dpi=180)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        (0.3, "Official bbox prompt"),
        (2.4, "ZIM ViT-B"),
        (4.4, "Horizontal flip TTA"),
        (6.7, "Guided filter r1"),
        (8.9, "Box-Support Prior"),
        (10.8, "Alpha / RGBA / Composer"),
    ]
    for x, label in boxes:
        ax.add_patch(plt.Rectangle((x, 1.05), 1.55, 0.9, facecolor="#EFF6FF", edgecolor="#2563EB", linewidth=1.4))
        ax.text(x + 0.775, 1.5, label, ha="center", va="center", fontsize=8.5, wrap=True)
    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + 1.55
        x1 = boxes[i + 1][0]
        ax.annotate("", xy=(x1 - 0.08, 1.5), xytext=(x0 + 0.08, 1.5), arrowprops=dict(arrowstyle="->", lw=1.4, color="#334155"))
    ax.text(6, 2.55, "PromptMatte-TTA-GF+BSP inference-only pipeline", ha="center", va="center", fontsize=14, weight="bold")
    ax.text(6, 0.45, "No training. No GT-derived inference. Benchmark metrics stay separate from Composer demos.", ha="center", va="center", fontsize=9, color="#475569")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def claim_safety_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4), dpi=180)
    ax.axis("off")
    ax.set_title("Claim Safety Boundary", fontsize=15, weight="bold", pad=12)
    allowed = [
        "valid706 official-prompt subset",
        "ZIM bbox baseline reproduction",
        "TTA-GF + BSP inference refinement",
        "Composer as application showcase",
    ]
    forbidden = [
        "Full MicroMat-3K claim",
        "Paper-level SOTA claim",
        "Demo metrics mixed into benchmark",
        "Approx Grad/Conn as official",
        "SAM2 result without teammate files",
    ]
    ax.add_patch(plt.Rectangle((0.03, 0.12), 0.43, 0.75, facecolor="#ECFDF5", edgecolor="#059669", linewidth=1.4))
    ax.add_patch(plt.Rectangle((0.54, 0.12), 0.43, 0.75, facecolor="#FEF2F2", edgecolor="#DC2626", linewidth=1.4))
    ax.text(0.245, 0.82, "Allowed", ha="center", va="center", fontsize=13, weight="bold", color="#047857")
    ax.text(0.755, 0.82, "Forbidden", ha="center", va="center", fontsize=13, weight="bold", color="#B91C1C")
    for i, item in enumerate(allowed):
        ax.text(0.07, 0.68 - i * 0.13, f"- {item}", ha="left", va="center", fontsize=10)
    for i, item in enumerate(forbidden):
        ax.text(0.58, 0.68 - i * 0.11, f"- {item}", ha="left", va="center", fontsize=10)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def find_sam2_candidates(root: Path) -> list[Path]:
    runs = root / "runs"
    candidates: dict[Path, float] = {}
    for pattern in ["*sam2*", "*sam3*", "*grounded*", "*segment*"]:
        for path in runs.glob(pattern):
            if path.is_dir():
                candidates[path] = path.stat().st_mtime
    return [p for p, _ in sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)]


def valid706_leaderboard() -> list[dict[str, Any]]:
    base_rows = {r.get("method", ""): r for r in read_csv(ZIM_RUN / "metrics/final_leaderboard.csv")}
    bsp_rows = {r.get("method", ""): r for r in read_csv(BSP_RUN / "metrics/all_valid706_box_prior_leaderboard.csv")}
    out = []
    for method in METHOD_ORDER:
        src = bsp_rows.get(method) if method == BSP else base_rows.get(method)
        if not src:
            continue
        n = int(safe_float(src.get("n"), 0))
        ok = int(safe_float(src.get("ok"), 0))
        out.append(
            {
                "method": method,
                "display_method": DISPLAY[method],
                "n_ok": ok,
                "SAD": safe_float(src.get("SAD")),
                "MSE": safe_float(src.get("MSE")),
                "MAE_x1000": safe_float(src.get("MAE_x1000")),
                "Boundary_SAD": safe_float(src.get("Boundary_SAD")),
                "failure_rate": safe_float(src.get("failure_rate"), 1.0 - ok / max(1, n)),
                "notes": "valid706 official-prompt subset; local SAD/MSE/MAE/Boundary metrics",
            }
        )
    return out


def holdout_leaderboard() -> list[dict[str, Any]]:
    holdout_ids = {r["sample_id"] for r in read_csv(BSP_RUN / "manifests/final_holdout506.csv")}
    all_metrics = read_csv(ZIM_RUN / "metrics/final_metrics_all.csv")
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in all_metrics:
        if row.get("sample_id") in holdout_ids and row.get("method") in [BASELINE, FLIP_TTA, GUIDED, TTA_GF]:
            by_method[row["method"]].append(row)
    out = []
    for method in [BASELINE, FLIP_TTA, GUIDED, TTA_GF]:
        rows = by_method.get(method, [])
        ok = [r for r in rows if r.get("status", "ok") == "ok"]
        out.append(
            {
                "method": method,
                "display_method": DISPLAY[method],
                "n_ok": len(ok),
                "SAD": mean_metric(rows, "SAD"),
                "MSE": mean_metric(rows, "MSE"),
                "MAE_x1000": mean_metric(rows, "MAE_x1000"),
                "Boundary_SAD": mean_metric(rows, "Boundary_SAD"),
                "failure_rate": 1.0 - len(ok) / max(1, len(holdout_ids)),
                "notes": "computed from final_metrics_all over BSP final_holdout506 sample_ids",
            }
        )
    bsp_hold = {r.get("method", ""): r for r in read_csv(BSP_RUN / "metrics/final_holdout506_box_prior_leaderboard.csv")}
    src = bsp_hold.get(BSP)
    if src:
        out.append(
            {
                "method": BSP,
                "display_method": DISPLAY[BSP],
                "n_ok": int(safe_float(src.get("ok"), 0)),
                "SAD": safe_float(src.get("SAD")),
                "MSE": safe_float(src.get("MSE")),
                "MAE_x1000": safe_float(src.get("MAE_x1000")),
                "Boundary_SAD": safe_float(src.get("Boundary_SAD")),
                "failure_rate": safe_float(src.get("failure_rate"), 0.0),
                "notes": "BSP fast full eval on same final_holdout506 manifest",
            }
        )
    return out


def improvement_rows(valid_rows: list[dict[str, Any]], holdout_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for split, rows in [("valid706", valid_rows), ("final_holdout506", holdout_rows)]:
        by = {r["method"]: r for r in rows}
        pairs = [
            (TTA_GF, BASELINE, "PromptMatte-TTA-GF vs ZIM bbox baseline"),
            (BSP, BASELINE, "PromptMatte-TTA-GF+BSP vs ZIM bbox baseline"),
            (BSP, TTA_GF, "BSP increment over PromptMatte-TTA-GF"),
        ]
        for left, right, label in pairs:
            if left not in by or right not in by:
                continue
            for metric in METRICS:
                lv = safe_float(by[left].get(metric))
                rv = safe_float(by[right].get(metric))
                imp = (rv - lv) / rv if not math.isnan(lv) and not math.isnan(rv) and abs(rv) > 1e-12 else math.nan
                out.append(
                    {
                        "split": split,
                        "comparison": label,
                        "left_method": left,
                        "right_method": right,
                        "metric": metric,
                        "right_value": rv,
                        "left_value": lv,
                        "relative_improvement": imp,
                        "relative_improvement_percent": imp * 100 if not math.isnan(imp) else math.nan,
                    }
                )
    return out


def result_inventory(run_dir: Path, sam_candidates: list[Path]) -> list[dict[str, Any]]:
    files = [
        ("ZIM final run", ZIM_RUN, ZIM_RUN / "metrics/final_leaderboard.csv"),
        ("ZIM per-sample metrics", ZIM_RUN, ZIM_RUN / "metrics/final_metrics_all.csv"),
        ("BSP valid706", BSP_RUN, BSP_RUN / "metrics/all_valid706_box_prior_leaderboard.csv"),
        ("BSP final_holdout506", BSP_RUN, BSP_RUN / "metrics/final_holdout506_box_prior_leaderboard.csv"),
        ("Composer showcase", COMPOSER_RUN, COMPOSER_RUN / "FINAL_COMPOSER_SHOWCASE.md"),
        ("Composer assets", COMPOSER_RUN, COMPOSER_RUN / "ppt_assets/composer_assets.csv"),
    ]
    rows = []
    for name, source_run, file_path in files:
        rows.append(
            {
                "item": name,
                "source_run": str(source_run),
                "file": str(file_path),
                "status": "FOUND" if file_path.exists() else "MISSING",
                "detail": f"{file_path.stat().st_size} bytes" if file_path.exists() else "required source not found",
            }
        )
    if sam_candidates:
        cand = sam_candidates[0]
        rows.append(
            {
                "item": "SAM2 teammate candidate",
                "source_run": str(cand),
                "file": str(cand),
                "status": "CANDIDATE_NOT_ALIGNED",
                "detail": "latest matching run exists, but no valid706 teammate leaderboard/metrics_all was found",
            }
        )
    else:
        rows.append(
            {
                "item": "SAM2 teammate result",
                "source_run": "",
                "file": "",
                "status": "MISSING",
                "detail": "no sam2/sam3/grounded/segment run directory found",
            }
        )
    write_csv(run_dir / "metrics/result_inventory.csv", rows)
    text = "# Result Inventory\n\n" + md_table(rows, ["item", "status", "source_run", "file", "detail"], None)
    write_text(run_dir / "reports/result_inventory.md", text)
    return rows


def sam2_reports(run_dir: Path, sam_candidates: list[Path], valid_rows: list[dict[str, Any]]) -> None:
    candidate = sam_candidates[0] if sam_candidates else None
    candidate_text = str(candidate) if candidate else "None"
    todo = f"""# SAM2 Result Missing TODO

No aligned SAM2 teammate benchmark result is available in this final package.

Latest auto-detected SAM-related candidate:

`{candidate_text}`

This candidate is not accepted as teammate benchmark data because it does not provide a 706-row aligned result table over the valid706 manifest. The detected Grounded-SAM2 text branch is blocked before actual text inference and cannot be used as SAM2 benchmark evidence.

## Files the teammate must provide

1. `metrics_all.csv`
   - Per-sample rows.
   - Required columns: `sample_id,method,status,SAD,MSE,MAE_x1000,Boundary_SAD,alpha_path_pred`.
   - Must use the exact `sample_id` values from `valid706_manifest_for_alignment.csv`.
2. `leaderboard.csv`
   - Required columns: `method,n,ok,failure_rate,SAD,MSE,MAE_x1000,Boundary_SAD`.
   - Include rows such as `sam2_bbox_binary` and/or `sam2_guided`.
3. `outputs/{{method}}/{{sample_id}}/alpha.png`
   - One alpha image per method/sample.
4. `manifest.csv`
   - The manifest actually used for inference.
5. `method_notes.md`
   - Prompt type, checkpoint, post-processing, and whether any samples failed.

## Alignment rules

- Do not use GT alpha for inference.
- Use the official bbox prompt protocol where applicable.
- If `n_ok != 706`, provide per-method `n_ok` and the missing/failed `sample_id` list.
- Composer/demo images must not be mixed into benchmark metrics.
- Approximate Grad/Conn, if any, must not be labeled official.
"""
    write_text(run_dir / "sam2_integration/SAM2_RESULT_MISSING_TODO.md", todo)
    write_text(run_dir / "SAM2_RESULT_MISSING_TODO.md", todo)
    report = f"""# SAM2 Integration Report

Status: `MISSING_ALIGNED_TEAMMATE_RESULT`

Auto-search looked for run directories containing `sam2`, `sam3`, `grounded`, or `segment`.

- latest candidate: `{candidate_text}`
- accepted into final leaderboard: `no`
- reason: no aligned valid706 `metrics_all.csv` + `leaderboard.csv` pair was found

## Current final methods already aligned

{md_table(valid_rows, ["method", "display_method", "n_ok", "SAD", "MSE", "MAE_x1000", "Boundary_SAD"], None)}

## Teammate action

Use the desktop package generated from this run. It contains the exact valid706/final_holdout506 manifests, final leaderboard, schema files, and a validator script.
"""
    write_text(run_dir / "sam2_integration/sam2_integration_report.md", report)
    write_csv(
        run_dir / "metrics/final_leaderboard_common_subset.csv",
        [
            {
                "status": "not_applicable",
                "reason": "SAM2 aligned result missing; common subset table will be generated after teammate provides metrics_all.csv.",
            }
        ],
        ["status", "reason"],
    )


def composer_assets(run_dir: Path) -> None:
    clean_link_or_copy(COMPOSER_RUN / "visuals/overview_contact_sheet.jpg", run_dir / "composer/overview_contact_sheet.jpg")
    clean_link_or_copy(COMPOSER_RUN / "ppt_assets/composer_assets.csv", run_dir / "composer/composer_assets.csv")
    clean_link_or_copy(COMPOSER_RUN / "visuals/overview_contact_sheet.jpg", run_dir / "ppt_assets/06_composer_showcase/overview_contact_sheet.jpg")

    sample_dirs = sorted([p for p in (COMPOSER_RUN / "visuals").iterdir() if p.is_dir()])
    gallery = ["<!doctype html><html><head><meta charset='utf-8'><title>PromptMatte Composer Gallery</title></head><body>"]
    gallery.append("<h1>PromptMatte Composer Showcase</h1>")
    gallery.append("<p>Application-only outputs: RGBA, replacement backgrounds, blur background, and alpha-edge visualization. These are not benchmark metrics.</p>")
    top_done = False
    for sdir in sample_dirs:
        dst_dir = run_dir / "composer/samples" / sdir.name
        for name in ["rgba.png", "replace_blue.jpg", "replace_white.jpg", "replace_gradient.jpg", "blur_bg.jpg", "alpha_edge.jpg", "contact_sheet.jpg", "original.jpg"]:
            clean_link_or_copy(sdir / name, dst_dir / name)
        if not top_done:
            for name in ["rgba.png", "replace_blue.jpg", "replace_white.jpg", "replace_gradient.jpg", "blur_bg.jpg", "alpha_edge.jpg"]:
                clean_link_or_copy(sdir / name, run_dir / "composer" / name)
                clean_link_or_copy(sdir / name, run_dir / "ppt_assets/06_composer_showcase" / name)
            top_done = True
        gallery.append(f"<h2>{html.escape(sdir.name)}</h2>")
        rel = f"samples/{html.escape(sdir.name)}/contact_sheet.jpg"
        gallery.append(f"<img src='{rel}' style='max-width:960px;width:95%;border:1px solid #ddd'>")
    gallery.append("</body></html>")
    write_text(run_dir / "composer/composer_gallery.html", "\n".join(gallery))
    summary = """# Composer Showcase Summary

PromptMatte Composer is an application/presentation module built on top of the predicted alpha matte.

Included outputs:

- RGBA foreground export
- blue/white/gradient background replacement
- blur-background composite
- alpha-edge visualization
- overview contact sheet

Claim boundary: Composer demonstrates usability of the alpha output. It does not change SAD/MSE/MAE/Boundary metrics and is not included in benchmark tables.
"""
    write_text(run_dir / "reports/composer_showcase_summary.md", summary)


def ppt_assets(run_dir: Path, valid_rows: list[dict[str, Any]], imp_rows: list[dict[str, Any]]) -> None:
    method_diagram(run_dir / "ppt_assets/01_method_diagram.png")
    table_image(
        run_dir / "ppt_assets/02_final_leaderboard.png",
        valid_rows,
        ["display_method", "n_ok", "SAD", "MSE", "MAE_x1000", "Boundary_SAD"],
        "Final Leaderboard on valid706",
        max_rows=8,
    )
    ablation = read_csv(ZIM_RUN / "metrics/fair_ablation_table.csv")
    table_image(
        run_dir / "ppt_assets/03_ablation_table.png",
        ablation,
        ["ablation", "display_method", "SAD", "MSE", "MAE_x1000", "Boundary_SAD"],
        "Ablation Table",
        max_rows=6,
    )
    clean_link_or_copy(
        ZIM_RUN / "ppt_assets/top_improvements_contact_sheet.png",
        run_dir / "ppt_assets/04_improvement_cases/top_improvements_contact_sheet.png",
    )
    clean_link_or_copy(
        ZIM_RUN / "ppt_assets/top_regressions_contact_sheet.png",
        run_dir / "ppt_assets/05_failure_cases/top_regressions_contact_sheet.png",
    )
    claim_safety_image(run_dir / "ppt_assets/07_claim_safety.png")
    table_image(
        run_dir / "ppt_assets/08_method_improvements.png",
        [r for r in imp_rows if r["split"] == "valid706" and r["comparison"] != "BSP increment over PromptMatte-TTA-GF"],
        ["comparison", "metric", "relative_improvement_percent"],
        "Main Improvements vs ZIM Baseline",
        max_rows=8,
    )


def claim_safety(run_dir: Path) -> None:
    allowed = """# Final Allowed Claims

- Dataset/protocol: MicroMat official-prompt valid706 subset, not full MicroMat-3K.
- Baseline: reproduced ZIM ViT-B official bbox prompt baseline on the same valid706 subset.
- Main method: PromptMatte-TTA-GF+BSP is an inference-only backend refinement.
- PromptMatte-TTA-GF means ZIM ViT-B official bbox prompt + horizontal flip TTA + guided filter r1.
- BSP means Box-Support Prior: a no-GT spatial support prior derived from the official bbox prompt.
- Metrics reported in the final leaderboard are local SAD, MSE, MAE_x1000, and Boundary_SAD on fixed sample sets.
- Composer is an application showcase for RGBA, background replacement, blur background, and alpha-edge visualization.
"""
    forbidden = """# Final Forbidden Claims

- Do not claim full MicroMat-3K evaluation.
- Do not claim paper-level SOTA.
- Do not claim official full Grad/Conn metrics; only bounded/partial official checks and approximate diagnostics exist.
- Do not mix Composer/demo outputs into benchmark metrics.
- Do not claim SAM2 teammate results until aligned `metrics_all.csv`, `leaderboard.csv`, outputs, manifest, and notes are provided.
- Do not imply GT alpha was used for inference.
- Do not present failed MS-TTA, risk gate, TTA-GF++, or LHR variants as the final method.
"""
    write_text(run_dir / "claim_safety/FINAL_ALLOWED_CLAIMS.md", allowed)
    write_text(run_dir / "claim_safety/FINAL_FORBIDDEN_CLAIMS.md", forbidden)


def final_summary(run_dir: Path, valid_rows: list[dict[str, Any]], holdout_rows: list[dict[str, Any]], imp_rows: list[dict[str, Any]]) -> None:
    valid_main = [r for r in valid_rows if r["method"] in [BASELINE, TTA_GF, BSP]]
    bsp_inc = [r for r in imp_rows if r["split"] == "valid706" and r["comparison"] == "BSP increment over PromptMatte-TTA-GF"]
    text = f"""# Final Submission Summary

Generated at: `{time.strftime('%Y-%m-%d %H:%M:%S')}`

## 1. Dataset and protocol

All benchmark numbers in this package use the MicroMat official-prompt valid706 subset. This is not full MicroMat-3K. The fixed final holdout table uses the 506-row `final_holdout506` manifest from the BSP run.

## 2. Baselines

The primary baseline is ZIM ViT-B with official bbox prompt. Additional fair components include horizontal flip TTA and guided filtering.

{md_table(valid_rows, ["method", "display_method", "n_ok", "SAD", "MSE", "MAE_x1000", "Boundary_SAD", "failure_rate"], None)}

## 3. PromptMatte-TTA-GF

PromptMatte-TTA-GF = ZIM ViT-B official bbox prompt + horizontal flip TTA + guided filter r1. It is the stable main backend before BSP.

## 4. Box-Support Prior

BSP = Box-Support Prior. It uses the official bbox as a no-GT spatial support prior to conservatively suppress alpha leakage outside the prompt support.

## 5. Results

Core valid706 comparison:

{md_table(valid_main, ["method", "display_method", "n_ok", "SAD", "MSE", "MAE_x1000", "Boundary_SAD"], None)}

Final holdout comparison:

{md_table(holdout_rows, ["method", "display_method", "n_ok", "SAD", "MSE", "MAE_x1000", "Boundary_SAD"], None)}

## 6. Ablation

The ablation table is exported to `ppt_assets/03_ablation_table.png` and `tables/ablation_table.csv`. Failed MS-TTA/risk/LHR/TTA-GF++ explorations are not used as final methods.

Selected improvement rows:

{md_table([r for r in imp_rows if r["split"] == "valid706"], ["comparison", "metric", "right_value", "left_value", "relative_improvement_percent"], None)}

## 7. Composer Showcase

Composer assets are application-only outputs: RGBA, background replacement, blur background, and alpha-edge visualization. They are copied/linked under `composer/` and `ppt_assets/06_composer_showcase/`.

## 8. SAM2 teammate integration

No aligned SAM2 teammate result was found. See `sam2_integration/SAM2_RESULT_MISSING_TODO.md` and the desktop teammate package for exact required files and schemas.

## 9. Failure and limitations

- valid706 subset only.
- BSP gives a small but consistent incremental improvement over PromptMatte-TTA-GF.
- Composer is not a benchmark method.
- SAM2 result is pending teammate submission.

## 10. Claim safety

Allowed and forbidden claims are written in `claim_safety/FINAL_ALLOWED_CLAIMS.md` and `claim_safety/FINAL_FORBIDDEN_CLAIMS.md`.

## 11. Reproduction

Run `bash repro/commands_reproduce_final_package.sh` from this package, or run `python3 scripts/final_submission/final_submission_package.py` from `/home/lpy/anisorisk/computer_vison`.
"""
    write_text(run_dir / "reports/FINAL_SUBMISSION_SUMMARY.md", text)
    write_text(run_dir / "FINAL_SUBMISSION_SUMMARY.md", text)
    write_csv(run_dir / "tables/final_leaderboard_valid706.csv", valid_rows)
    write_csv(run_dir / "tables/final_leaderboard_final_holdout506.csv", holdout_rows)
    shutil.copy2(ZIM_RUN / "metrics/fair_ablation_table.csv", run_dir / "tables/ablation_table.csv")


def teammate_package(run_dir: Path) -> None:
    pkg = run_dir / "sam2_integration/teammate_sam2_alignment_package"
    pkg.mkdir(parents=True, exist_ok=True)
    files = [
        (BSP_RUN / "manifests/all_valid706.csv", pkg / "valid706_manifest_for_alignment.csv"),
        (BSP_RUN / "manifests/final_holdout506.csv", pkg / "final_holdout506_manifest_for_alignment.csv"),
        (run_dir / "metrics/final_leaderboard_valid706.csv", pkg / "current_final_leaderboard_valid706.csv"),
        (run_dir / "metrics/final_leaderboard_final_holdout506.csv", pkg / "current_final_leaderboard_final_holdout506.csv"),
        (run_dir / "metrics/final_method_improvements.csv", pkg / "current_final_method_improvements.csv"),
        (run_dir / "sam2_integration/SAM2_RESULT_MISSING_TODO.md", pkg / "SAM2_RESULT_MISSING_TODO.md"),
        (run_dir / "claim_safety/FINAL_ALLOWED_CLAIMS.md", pkg / "FINAL_ALLOWED_CLAIMS.md"),
        (run_dir / "claim_safety/FINAL_FORBIDDEN_CLAIMS.md", pkg / "FINAL_FORBIDDEN_CLAIMS.md"),
        (run_dir / "reports/FINAL_SUBMISSION_SUMMARY.md", pkg / "FINAL_SUBMISSION_SUMMARY.md"),
    ]
    for src, dst in files:
        if src.exists():
            shutil.copy2(src, dst)
    sample_ids = [r["sample_id"] for r in read_csv(BSP_RUN / "manifests/all_valid706.csv")]
    write_text(pkg / "sample_id_list_valid706.txt", "\n".join(sample_ids) + "\n")
    hold_ids = [r["sample_id"] for r in read_csv(BSP_RUN / "manifests/final_holdout506.csv")]
    write_text(pkg / "sample_id_list_final_holdout506.txt", "\n".join(hold_ids) + "\n")
    write_csv(
        pkg / "sam2_expected_metrics_all_schema.csv",
        [
            {"column": "sample_id", "required": "yes", "note": "must match valid706_manifest_for_alignment.csv"},
            {"column": "method", "required": "yes", "note": "e.g. sam2_bbox_binary or sam2_guided"},
            {"column": "status", "required": "yes", "note": "ok or failure"},
            {"column": "SAD", "required": "yes", "note": "same metric protocol as final table"},
            {"column": "MSE", "required": "yes", "note": "same metric protocol as final table"},
            {"column": "MAE_x1000", "required": "yes", "note": "same metric protocol as final table"},
            {"column": "Boundary_SAD", "required": "yes", "note": "same metric protocol as final table"},
            {"column": "alpha_path_pred", "required": "yes", "note": "path to output alpha.png"},
        ],
    )
    write_csv(
        pkg / "sam2_expected_leaderboard_schema.csv",
        [
            {"column": "method", "required": "yes"},
            {"column": "n", "required": "yes"},
            {"column": "ok", "required": "yes"},
            {"column": "failure_rate", "required": "yes"},
            {"column": "SAD", "required": "yes"},
            {"column": "MSE", "required": "yes"},
            {"column": "MAE_x1000", "required": "yes"},
            {"column": "Boundary_SAD", "required": "yes"},
        ],
    )
    validator = """#!/usr/bin/env python3
import argparse, csv, sys
from pathlib import Path

def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='valid706_manifest_for_alignment.csv')
    ap.add_argument('--metrics-all', required=True)
    args = ap.parse_args()
    manifest_ids = [r['sample_id'] for r in read_csv(Path(args.manifest))]
    manifest_set = set(manifest_ids)
    rows = read_csv(Path(args.metrics_all))
    methods = sorted(set(r.get('method', '') for r in rows))
    ok = True
    print('manifest_n', len(manifest_ids))
    for method in methods:
        mrows = [r for r in rows if r.get('method') == method]
        ids = set(r.get('sample_id') for r in mrows if r.get('status', 'ok') == 'ok')
        extra = ids - manifest_set
        missing = manifest_set - ids
        print(method, 'rows', len(mrows), 'ok_ids', len(ids), 'missing', len(missing), 'extra', len(extra))
        if extra:
            ok = False
            print('extra_sample_ids', sorted(list(extra))[:20])
        if len(ids) != len(manifest_set):
            ok = False
            print('missing_sample_ids', sorted(list(missing))[:20])
    if not ok:
        sys.exit(2)

if __name__ == '__main__':
    main()
"""
    write_text(pkg / "validate_sam2_submission.py", validator)
    os.chmod(pkg / "validate_sam2_submission.py", 0o755)
    readme = """# PromptMatte SAM2 Teammate Alignment Package

Use this folder to align SAM2 results with the final PromptMatte leaderboard.

## What to run

1. Use `valid706_manifest_for_alignment.csv` as the primary benchmark manifest.
2. Produce per-sample alpha files under `outputs/{method}/{sample_id}/alpha.png`.
3. Produce `metrics_all.csv` and `leaderboard.csv` using the schemas in this folder.
4. Validate sample alignment:

```bash
python3 validate_sam2_submission.py --metrics-all metrics_all.csv
```

## Important

- If `n_ok != 706`, report the failed/missing sample IDs.
- Do not use GT alpha for inference.
- Do not label approximate metrics as official.
- The current final table is in `current_final_leaderboard_valid706.csv`.
"""
    write_text(pkg / "README_FOR_TEAMMATE.md", readme)


def repro_script(run_dir: Path) -> None:
    script = f"""#!/usr/bin/env bash
set -euo pipefail
cd {ROOT}
python3 scripts/final_submission/final_submission_package.py
"""
    path = run_dir / "repro/commands_reproduce_final_package.sh"
    write_text(path, script)
    os.chmod(path, 0o755)
    result = subprocess.run(["bash", "-n", str(path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    write_text(run_dir / "logs/repro_bash_n.log", result.stdout + f"\nreturncode={result.returncode}\n")
    if result.returncode != 0:
        raise RuntimeError("repro script failed bash -n")


def run(args: argparse.Namespace) -> Path:
    run_id = args.run_id or f"final_submission_package_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    ensure_dir_tree(run_dir)

    latest = ROOT / "runs/latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(run_id)

    valid_rows = valid706_leaderboard()
    holdout_rows = holdout_leaderboard()
    imp_rows = improvement_rows(valid_rows, holdout_rows)
    sam_candidates = find_sam2_candidates(ROOT)

    write_csv(run_dir / "metrics/final_leaderboard_valid706.csv", valid_rows)
    write_csv(run_dir / "metrics/final_leaderboard_final_holdout506.csv", holdout_rows)
    write_csv(run_dir / "metrics/final_method_improvements.csv", imp_rows)
    write_csv(
        run_dir / "metrics/per_method_n_ok.csv",
        [
            {"split": "valid706", "method": r["method"], "n_ok": r["n_ok"], "failure_rate": r["failure_rate"]}
            for r in valid_rows
        ]
        + [
            {"split": "final_holdout506", "method": r["method"], "n_ok": r["n_ok"], "failure_rate": r["failure_rate"]}
            for r in holdout_rows
        ],
    )

    result_inventory(run_dir, sam_candidates)
    sam2_reports(run_dir, sam_candidates, valid_rows)
    composer_assets(run_dir)
    claim_safety(run_dir)
    ppt_assets(run_dir, valid_rows, imp_rows)
    final_summary(run_dir, valid_rows, holdout_rows, imp_rows)
    teammate_package(run_dir)
    repro_script(run_dir)

    status = f"FINAL SUBMISSION PACKAGE DONE at {time.strftime('%Y-%m-%d %H:%M:%S')}\nRUN_DIR={run_dir}\n"
    write_text(run_dir / "DONE.txt", status)
    print(status.strip())
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
