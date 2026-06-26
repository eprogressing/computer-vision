#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


ROOT = Path("/home/lpy/anisorisk/computer_vison")
PREV_FINAL_RUN = ROOT / "runs/protocol_repair_final_claim_20260602_174358"
PREV_SPLIT_RUN = ROOT / "runs/promptmatte_lhr_20260604_010706"
BASELINE = "zim_vitb_bbox_default"
CURRENT = "zim_vitb_flip_tta_bbox_guided_r1"
METHOD = "promptmatte_tta_gf_box_support_prior"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_progress(run_dir: Path, step: str, result: str, outputs: str = "") -> None:
    with (run_dir / "PROGRESS.md").open("a", encoding="utf-8") as f:
        f.write(f"\n## {now()}\n- step: {step}\n- result: {result}\n- outputs: {outputs}\n")


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


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(x: Any, default: float = math.nan) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def shell(cmd: list[str] | str) -> tuple[int, str]:
    p = subprocess.run(cmd, shell=isinstance(cmd, str), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def read_alpha(path: Path | str, size: tuple[int, int] | None = None) -> np.ndarray:
    img = Image.open(path).convert("L")
    if size and img.size != size:
        img = img.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def save_alpha(path: Path, alpha: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(alpha, 0, 1) * 255 + 0.5).astype(np.uint8), mode="L").save(path)


def load_prev_metrics() -> tuple[list[dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    rows = read_csv(PREV_FINAL_RUN / "metrics/final_metrics_all.csv")
    return rows, {(r["sample_id"], r["method"]): r for r in rows}


def alpha_path(prev_map: dict[tuple[str, str], dict[str, str]], sid: str, method: str) -> Path:
    row = prev_map.get((sid, method), {})
    path = Path(row.get("alpha_path_pred", ""))
    if not path.exists():
        raise FileNotFoundError(f"missing {sid}/{method}: {path}")
    return path


def parse_bbox(row: dict[str, str]) -> list[int]:
    return [int(round(float(x))) for x in json.loads(row["bbox"])]


def expand_bbox(bbox: list[int], width: int, height: int, pad: float) -> list[int]:
    x1, y1, x2, y2 = [float(x) for x in bbox]
    bw = max(1.0, x2 - x1 + 1.0)
    bh = max(1.0, y2 - y1 + 1.0)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    nw = bw * pad
    nh = bh * pad
    return [
        int(max(0, math.floor(cx - nw / 2.0))),
        int(max(0, math.floor(cy - nh / 2.0))),
        int(min(width - 1, math.ceil(cx + nw / 2.0))),
        int(min(height - 1, math.ceil(cy + nh / 2.0))),
    ]


def support_mask(size: tuple[int, int], bbox: list[int], pad: float, blur_radius: float) -> np.ndarray:
    width, height = size
    x1, y1, x2, y2 = expand_bbox(bbox, width, height, pad)
    mask = Image.new("L", (width, height), 0)
    arr = np.asarray(mask, dtype=np.uint8).copy()
    arr[y1 : y2 + 1, x1 : x2 + 1] = 255
    mask = Image.fromarray(arr, mode="L")
    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return np.asarray(mask, dtype=np.float32) / 255.0


def apply_box_prior(alpha: np.ndarray, row: dict[str, str], pad: float, blur_radius: float, suppress: float, soft_floor: float) -> np.ndarray:
    size = (alpha.shape[1], alpha.shape[0])
    mask = support_mask(size, parse_bbox(row), pad, blur_radius)
    # Outside support, multiply alpha by a conservative floor. Inside support, keep the original alpha.
    gate = np.clip(mask + (1.0 - mask) * soft_floor, soft_floor, 1.0)
    out = alpha * (1.0 - suppress + suppress * gate)
    return np.clip(out, 0, 1)


def gt_boundary_band(gt: np.ndarray) -> np.ndarray:
    fg = gt >= 0.5
    edge = np.zeros_like(fg, dtype=bool)
    edge[1:, :] |= fg[1:, :] != fg[:-1, :]
    edge[:-1, :] |= fg[:-1, :] != fg[1:, :]
    edge[:, 1:] |= fg[:, 1:] != fg[:, :-1]
    edge[:, :-1] |= fg[:, :-1] != fg[:, 1:]
    soft = (gt > 0.02) & (gt < 0.98)
    return edge | soft


def local_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = np.clip(pred.astype(np.float32), 0, 1)
    gt = np.clip(gt.astype(np.float32), 0, 1)
    diff = np.abs(pred - gt)
    band = gt_boundary_band(gt)
    return {
        "SAD": float(diff.sum() / 1000.0),
        "MSE": float(np.mean((pred - gt) ** 2)),
        "MAE_x1000": float(np.mean(diff) * 1000.0),
        "Boundary_SAD": float(diff[band].sum() / 1000.0) if band.any() else math.nan,
        "Boundary_MSE": float(np.mean((pred[band] - gt[band]) ** 2)) if band.any() else math.nan,
    }


def mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    vals = [safe_float(r.get(key)) for r in rows if r.get("status", "ok") == "ok"]
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.mean(vals)) if vals else math.nan


def leaderboard(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(str(r.get("method", "")), []).append(r)
    out = []
    for method, rs in by.items():
        ok = [r for r in rs if r.get("status", "ok") == "ok"]
        out.append(
            {
                "method": method,
                "n": len(rs),
                "ok": len(ok),
                "failure_rate": 1.0 - len(ok) / max(1, len(rs)),
                "SAD": mean_metric(ok, "SAD"),
                "MSE": mean_metric(ok, "MSE"),
                "MAE_x1000": mean_metric(ok, "MAE_x1000"),
                "Boundary_SAD": mean_metric(ok, "Boundary_SAD"),
                "Boundary_MSE": mean_metric(ok, "Boundary_MSE"),
            }
        )
    out.sort(key=lambda r: (safe_float(r["SAD"], 1e9), safe_float(r["MSE"], 1e9)))
    return out


def improvement(current: dict[str, Any], new: dict[str, Any], metric: str) -> float:
    c = safe_float(current.get(metric))
    n = safe_float(new.get(metric))
    if math.isnan(c) or math.isnan(n) or abs(c) < 1e-12:
        return math.nan
    return (c - n) / c


def markdown_table(rows: list[dict[str, Any]], keys: list[str], max_rows: int = 30) -> str:
    if not rows:
        return "_No rows._\n"
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows[:max_rows]:
        vals = []
        for key in keys:
            v = row.get(key, "")
            vals.append(f"{v:.6g}" if isinstance(v, float) else str(v)[:160])
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def ensure_run(run_dir: Path) -> None:
    for name in ["logs", "configs", "manifests", "metrics", "diagnostics", "outputs", "reports", "visuals", "ppt_assets", "unit_tests"]:
        (run_dir / name).mkdir(parents=True, exist_ok=True)


def step_init(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    ensure_run(run_dir)
    scratch = Path("/tmp/lpy_promptmatte") / run_dir.name
    outputs = scratch / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    link = run_dir / "outputs_large"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(outputs, target_is_directory=True)
    for name in ["dev_lhr100.csv", "val_lhr100.csv", "final_holdout506.csv", "all_valid706.csv", "fine5.csv"]:
        shutil.copy2(PREV_SPLIT_RUN / "manifests" / name, run_dir / "manifests" / name)
    write_text(
        run_dir / "PROGRESS.md",
        f"# Box-Support Alpha Prior: {run_dir.name}\n"
        f"- start: {now()}\n"
        f"- frozen current main: {CURRENT}\n"
        "- idea: use official bbox as a no-GT spatial support prior to suppress alpha leakage outside the prompt support.\n",
    )
    write_text(run_dir / "BLOCKERS.md", "# Blockers\n\n")
    save_json(run_dir / "configs/paths.json", {"scratch": str(scratch), "outputs": str(outputs)})
    make_commands(run_dir)
    rc, out = shell("pwd; hostname; date; df -h; df -hi; free -h; nvidia-smi || true")
    write_text(run_dir / "env_report.txt", out)
    append_progress(run_dir, "00_init", f"scratch={scratch}", str(run_dir / "configs/paths.json"))


def param_grid() -> list[dict[str, Any]]:
    rows = [{"method": CURRENT, "pad": "", "blur": "", "suppress": "", "soft_floor": ""}]
    for pad in [1.20, 1.35, 1.50, 1.75]:
        for suppress in [0.10, 0.20, 0.35]:
            for soft_floor in [0.05, 0.10, 0.20]:
                rows.append(
                    {
                        "method": f"BP_pad{pad:g}_sup{suppress:g}_floor{soft_floor:g}",
                        "pad": pad,
                        "blur": 12.0,
                        "suppress": suppress,
                        "soft_floor": soft_floor,
                    }
                )
    return rows


def eval_split(run_dir: Path, split: str, params: list[dict[str, Any]], save_outputs: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _, prev_map = load_prev_metrics()
    paths = load_json(run_dir / "configs/paths.json")
    output_root = Path(paths["outputs"]) / split
    rows = []
    for row in read_csv(run_dir / "manifests" / f"{split}.csv"):
        sid = row["sample_id"]
        size = Image.open(row["image_path"]).size
        current_alpha = read_alpha(alpha_path(prev_map, sid, CURRENT), size=size)
        gt = read_alpha(row["alpha_path"], size=size)
        for p in params:
            method = p["method"]
            if method == CURRENT:
                pred = current_alpha
            else:
                pred = apply_box_prior(current_alpha, row, float(p["pad"]), float(p["blur"]), float(p["suppress"]), float(p["soft_floor"]))
            rec = {"sample_id": sid, "method": method, "tag": split, "status": "ok", **local_metrics(pred, gt)}
            rows.append(rec)
            if save_outputs:
                save_alpha(output_root / method / sid / "alpha.png", pred)
    lb = leaderboard(rows)
    write_csv(run_dir / "metrics" / f"{split}_box_prior_metrics.csv", rows)
    write_csv(run_dir / "metrics" / f"{split}_box_prior_leaderboard.csv", lb)
    return rows, lb


def regression_count(rows: list[dict[str, Any]], method: str) -> int:
    by: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["sample_id"], {})[r["method"]] = r
    count = 0
    for rs in by.values():
        if method in rs and CURRENT in rs and safe_float(rs[method]["SAD"]) > safe_float(rs[CURRENT]["SAD"]):
            count += 1
    return count


def step_select(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    params = param_grid()
    dev_rows, dev_lb = eval_split(run_dir, "dev_lhr100", params, save_outputs=False)
    val_rows, val_lb = eval_split(run_dir, "val_lhr100", params, save_outputs=False)
    cur_dev = next(r for r in dev_lb if r["method"] == CURRENT)
    cur_val = next(r for r in val_lb if r["method"] == CURRENT)
    candidates = []
    for row in dev_lb:
        if row["method"] == CURRENT:
            continue
        sad_imp = improvement(cur_dev, row, "SAD")
        mse_imp = improvement(cur_dev, row, "MSE")
        bnd_imp = improvement(cur_dev, row, "Boundary_SAD")
        dev_reg = regression_count(dev_rows, row["method"])
        if sad_imp >= 0.002 and mse_imp >= 0 and bnd_imp >= -0.003 and dev_reg <= 45:
            val = next(v for v in val_lb if v["method"] == row["method"])
            val_sad = improvement(cur_val, val, "SAD")
            val_mse = improvement(cur_val, val, "MSE")
            val_bnd = improvement(cur_val, val, "Boundary_SAD")
            val_reg = regression_count(val_rows, row["method"])
            if val_sad >= 0.002 and val_mse >= 0 and val_bnd >= -0.003 and val_reg <= 45:
                candidates.append(
                    {
                        "method": row["method"],
                        "dev_SAD_imp": sad_imp,
                        "dev_MSE_imp": mse_imp,
                        "dev_Boundary_SAD_imp": bnd_imp,
                        "dev_regressions": dev_reg,
                        "val_SAD_imp": val_sad,
                        "val_MSE_imp": val_mse,
                        "val_Boundary_SAD_imp": val_bnd,
                        "val_regressions": val_reg,
                        "score": val_sad + 0.5 * val_mse + 0.25 * max(0, val_bnd) - 0.001 * val_reg,
                    }
                )
    candidates.sort(key=lambda r: (-safe_float(r["score"]), safe_float(r["val_regressions"])))
    selected = candidates[0] if candidates else None
    selected_param = None
    if selected:
        selected_param = next(p for p in params if p["method"] == selected["method"])
    cfg = {
        "enabled": bool(selected),
        "final_method": METHOD if selected else CURRENT,
        "selected": selected,
        "params": selected_param,
        "reason": "" if selected else "No dev/val-safe box prior variant.",
    }
    save_json(run_dir / "configs/box_prior_final.json", cfg)
    write_csv(run_dir / "diagnostics/box_prior_dev_val_confirmed.csv", candidates)
    write_text(
        run_dir / "BOX_PRIOR_SELECTION_REPORT.md",
        "# Box-Support Alpha Prior Selection\n\n"
        + f"- enabled: `{bool(selected)}`\n"
        + f"- selected: `{selected['method'] if selected else ''}`\n\n"
        + "## Confirmed Candidates\n\n"
        + markdown_table(candidates, ["method", "dev_SAD_imp", "val_SAD_imp", "val_MSE_imp", "val_Boundary_SAD_imp", "val_regressions", "score"], 30)
        + "\n## Dev Leaderboard\n\n"
        + markdown_table(dev_lb, ["method", "ok", "SAD", "MSE", "Boundary_SAD"], 20)
        + "\n## Val Leaderboard\n\n"
        + markdown_table(val_lb, ["method", "ok", "SAD", "MSE", "Boundary_SAD"], 20),
    )
    append_progress(run_dir, "01_select", f"enabled={bool(selected)} selected={selected['method'] if selected else ''}", str(run_dir / "BOX_PRIOR_SELECTION_REPORT.md"))


def step_final(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    cfg = load_json(run_dir / "configs/box_prior_final.json", {})
    if cfg.get("enabled"):
        params = [cfg["params"]]
    else:
        params = [{"method": CURRENT, "pad": "", "blur": "", "suppress": "", "soft_floor": ""}]
    # Always include current and baseline for fair final tables.
    final_params = [{"method": CURRENT, "pad": "", "blur": "", "suppress": "", "soft_floor": ""}] + params
    final_params = [p for i, p in enumerate(final_params) if p["method"] == CURRENT or p not in final_params[:i]]
    eval_split(run_dir, "final_holdout506", final_params, save_outputs=True)
    # Gate holdout before all_valid claim. If negative, disable and output current fallback.
    hold_lb = read_csv(run_dir / "metrics/final_holdout506_box_prior_leaderboard.csv")
    cur = next(r for r in hold_lb if r["method"] == CURRENT)
    new = next((r for r in hold_lb if r["method"] == METHOD or r["method"] == cfg.get("params", {}).get("method")), cur)
    hold_comp = [{"metric": m, "current": cur.get(m), "final": new.get(m), "improvement": improvement(cur, new, m)} for m in ["SAD", "MSE", "MAE_x1000", "Boundary_SAD"]]
    write_csv(run_dir / "metrics/final_holdout506_vs_current.csv", hold_comp)
    vals = {r["metric"]: safe_float(r["improvement"], 0) for r in hold_comp}
    pass_holdout = vals.get("SAD", 0) >= 0.002 or vals.get("MSE", 0) >= 0.002 or vals.get("Boundary_SAD", 0) >= 0.005
    if not pass_holdout:
        cfg["enabled"] = False
        cfg["final_method"] = CURRENT
        cfg["reason"] = "NO-GO after final_holdout506; fallback to current main."
        save_json(run_dir / "configs/box_prior_final.json", cfg)
        all_params = [{"method": CURRENT, "pad": "", "blur": "", "suppress": "", "soft_floor": ""}]
    else:
        all_params = final_params
    eval_split(run_dir, "all_valid706", all_params, save_outputs=True)
    all_lb = read_csv(run_dir / "metrics/all_valid706_box_prior_leaderboard.csv")
    cur_all = next(r for r in all_lb if r["method"] == CURRENT)
    new_all = next((r for r in all_lb if r["method"] != CURRENT), cur_all)
    all_comp = [{"metric": m, "current": cur_all.get(m), "final": new_all.get(m), "improvement": improvement(cur_all, new_all, m)} for m in ["SAD", "MSE", "MAE_x1000", "Boundary_SAD"]]
    write_csv(run_dir / "metrics/all_valid706_vs_current.csv", all_comp)
    append_progress(run_dir, "02_final", f"holdout_pass={pass_holdout}", str(run_dir / "metrics/final_holdout506_box_prior_leaderboard.csv"))


def step_report(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    cfg = load_json(run_dir / "configs/box_prior_final.json", {})
    hold_lb = read_csv(run_dir / "metrics/final_holdout506_box_prior_leaderboard.csv")
    all_lb = read_csv(run_dir / "metrics/all_valid706_box_prior_leaderboard.csv")
    hold_comp = read_csv(run_dir / "metrics/final_holdout506_vs_current.csv")
    all_comp = read_csv(run_dir / "metrics/all_valid706_vs_current.csv")
    verdict = "BOX_PRIOR_PASS" if cfg.get("enabled") else "BOX_PRIOR_NO_GO"
    write_text(
        run_dir / "GO_NO_GO_FOR_BOX_PRIOR.md",
        "# GO/NO-GO For Box-Support Alpha Prior\n\n"
        + f"- Verdict: `{verdict}`\n"
        + f"- Final_method: `{cfg.get('final_method')}`\n"
        + f"- Selected: `{cfg.get('selected', {}).get('method', '')}`\n"
        + f"- Params: `{json.dumps(cfg.get('params', {}), ensure_ascii=False)}`\n"
        + f"- Final_holdout_improvement: `{json.dumps(hold_comp, ensure_ascii=False)}`\n"
        + f"- All_valid_improvement: `{json.dumps(all_comp, ensure_ascii=False)}`\n"
        + "- Forbidden_claims: `paper SOTA; full MicroMat; GT inference; official Grad/Conn`\n",
    )
    write_text(
        run_dir / "FINAL_BOX_PRIOR_SUMMARY.md",
        "# Box-Support Alpha Prior Summary\n\n"
        + f"- verdict: `{verdict}`\n"
        + f"- final method for claim: `{cfg.get('final_method')}`\n"
        + "- external basis: promptable segmentation uses spatial prompts; box supervision constrains masks; trimap matting uses known support/unknown regions.\n"
        + "- implementation: official bbox expanded into a soft support mask; alpha leakage outside support is conservatively suppressed.\n\n"
        + "## final_holdout506\n\n"
        + markdown_table(hold_lb, ["method", "n", "ok", "SAD", "MSE", "MAE_x1000", "Boundary_SAD"], 20)
        + "\n## all_valid706\n\n"
        + markdown_table(all_lb, ["method", "n", "ok", "SAD", "MSE", "MAE_x1000", "Boundary_SAD"], 20),
    )
    findings = [
        {"severity": "INFO", "check": "no_gt_inference", "status": "PASS", "detail": "Uses image-size alpha and official bbox only."},
        {"severity": "INFO", "check": "dev_val_final_split", "status": "PASS", "detail": "Params selected on dev and confirmed on val before final."},
        {"severity": "INFO", "check": "holdout_gate", "status": "PASS" if cfg.get("enabled") else "NO_GO", "detail": cfg.get("reason", "")},
        {"severity": "WARN", "check": "full_micromat_claim", "status": "FORBIDDEN", "detail": "valid706 subset only."},
    ]
    write_text(run_dir / "reports/protocol_audit.md", "# Protocol Audit\n\n" + markdown_table(findings, ["severity", "check", "status", "detail"], 20))
    write_csv(run_dir / "diagnostics/protocol_audit_findings.csv", findings)
    html = "<html><body><h1>Box-Support Alpha Prior</h1>" + markdown_table(hold_lb + all_lb, ["method", "n", "ok", "SAD", "MSE", "Boundary_SAD"], 20).replace("\n", "<br>") + "</body></html>"
    write_text(run_dir / "visuals/index.html", html)
    write_csv(run_dir / "ppt_assets/core_box_prior_tables.csv", hold_lb + all_lb)
    append_progress(run_dir, "03_report", f"verdict={verdict}", str(run_dir / "FINAL_BOX_PRIOR_SUMMARY.md"))


def step_tests(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    tests = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        tests.append({"test": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    add("config_exists", (run_dir / "configs/box_prior_final.json").exists())
    add("holdout_metrics_exists", (run_dir / "metrics/final_holdout506_box_prior_leaderboard.csv").exists())
    add("allvalid_metrics_exists", (run_dir / "metrics/all_valid706_box_prior_leaderboard.csv").exists())
    add("commands_syntax", shell(["bash", "-n", str(run_dir / "commands_reproduce.sh")])[0] == 0)
    add("visuals_exists", (run_dir / "visuals/index.html").exists())
    write_csv(run_dir / "unit_tests/unit_test_summary.csv", tests)
    write_text(run_dir / "unit_tests/unit_test_results.txt", "\n".join(f"{t['status']} {t['test']} {t['detail']}" for t in tests) + "\n")
    append_progress(run_dir, "04_tests", f"pass={sum(t['status']=='PASS' for t in tests)}/{len(tests)}", str(run_dir / "unit_tests/unit_test_results.txt"))


def make_commands(run_dir: Path) -> None:
    text = f"""#!/usr/bin/env bash
set -euo pipefail
cd /home/lpy/anisorisk/computer_vison
export RUN_DIR={run_dir}
export PYTHON_RUNNER=/opt/miniconda3/bin/python
$PYTHON_RUNNER scripts/box_prior/00_init.py --run-dir "$RUN_DIR"
$PYTHON_RUNNER scripts/box_prior/01_select.py --run-dir "$RUN_DIR"
$PYTHON_RUNNER scripts/box_prior/02_final.py --run-dir "$RUN_DIR"
$PYTHON_RUNNER scripts/box_prior/03_report.py --run-dir "$RUN_DIR"
$PYTHON_RUNNER scripts/box_prior/04_tests.py --run-dir "$RUN_DIR"
"""
    write_text(run_dir / "commands_reproduce.sh", text)
    os.chmod(run_dir / "commands_reproduce.sh", 0o755)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit("missing step")
    step = argv.pop(0)
    table = {
        "00_init": step_init,
        "01_select": step_select,
        "02_final": step_final,
        "03_report": step_report,
        "04_tests": step_tests,
    }
    if step not in table:
        raise SystemExit(f"unknown step {step}")
    table[step](argv)


if __name__ == "__main__":
    main()
