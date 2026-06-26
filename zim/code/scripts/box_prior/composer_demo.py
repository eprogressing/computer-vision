#!/usr/bin/env python
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path("/home/lpy/anisorisk/computer_vison")
PREV_FINAL_RUN = ROOT / "runs/protocol_repair_final_claim_20260602_174358"
PREV_SPLIT_RUN = ROOT / "runs/promptmatte_lhr_20260604_010706"
CURRENT = "zim_vitb_flip_tta_bbox_guided_r1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def alpha_path_map() -> dict[tuple[str, str], Path]:
    rows = read_csv(PREV_FINAL_RUN / "metrics/final_metrics_all.csv")
    return {(r["sample_id"], r["method"]): Path(r["alpha_path_pred"]) for r in rows}


def read_alpha(path: Path, size: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("L")
    if img.size != size:
        img = img.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def feather_alpha(alpha: np.ndarray, radius: float = 1.2) -> np.ndarray:
    img = Image.fromarray((np.clip(alpha, 0, 1) * 255 + 0.5).astype(np.uint8), mode="L")
    soft = np.asarray(img.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32) / 255.0
    band = (alpha > 0.02) & (alpha < 0.98)
    out = alpha.copy()
    out[band] = soft[band]
    return np.clip(out, 0, 1)


def composite(rgb: Image.Image, alpha: np.ndarray, bg: Image.Image) -> Image.Image:
    fg = np.asarray(rgb.convert("RGB"), dtype=np.float32)
    bg_arr = np.asarray(bg.resize(rgb.size).convert("RGB"), dtype=np.float32)
    a = alpha[..., None]
    out = fg * a + bg_arr * (1.0 - a)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def make_bg(size: tuple[int, int], kind: str, rgb: Image.Image) -> Image.Image:
    w, h = size
    if kind == "white":
        return Image.new("RGB", size, (245, 246, 248))
    if kind == "blue":
        return Image.new("RGB", size, (42, 96, 180))
    if kind == "warm":
        return Image.new("RGB", size, (224, 190, 122))
    if kind == "blur":
        return rgb.convert("RGB").filter(ImageFilter.GaussianBlur(radius=max(8, min(w, h) // 40)))
    grad = Image.new("RGB", size)
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        t = y / max(1, h - 1)
        arr[y, :, 0] = int(34 * (1 - t) + 220 * t)
        arr[y, :, 1] = int(98 * (1 - t) + 230 * t)
        arr[y, :, 2] = int(155 * (1 - t) + 210 * t)
    return Image.fromarray(arr, mode="RGB")


def edge_overlay(rgb: Image.Image, alpha: np.ndarray) -> Image.Image:
    base = np.asarray(rgb.convert("RGB"), dtype=np.uint8).copy()
    band = (alpha > 0.03) & (alpha < 0.97)
    base[band] = (255, 64, 64)
    return Image.fromarray(base, mode="RGB")


def fit_thumb(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, (248, 248, 248))
    tmp = img.convert("RGB")
    tmp.thumbnail((size[0] - 8, size[1] - 28))
    canvas.paste(tmp, ((size[0] - tmp.width) // 2, 4))
    return canvas


def contact_sheet(items: list[tuple[str, Image.Image]], out: Path, cols: int = 4, thumb: tuple[int, int] = (260, 220)) -> None:
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb[0], rows * thumb[1]), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, (label, img) in enumerate(items):
        x = (i % cols) * thumb[0]
        y = (i // cols) * thumb[1]
        tile = fit_thumb(img, thumb)
        sheet.paste(tile, (x, y))
        draw.text((x + 8, y + thumb[1] - 22), label[:32], fill=(20, 20, 20))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def main() -> None:
    out_root = ROOT / "runs/promptmatte_composer_showcase_20260604"
    (out_root / "visuals").mkdir(parents=True, exist_ok=True)
    (out_root / "ppt_assets").mkdir(parents=True, exist_ok=True)
    amap = alpha_path_map()
    rows = read_csv(PREV_SPLIT_RUN / "manifests/fine5.csv") + read_csv(PREV_SPLIT_RUN / "manifests/dev_lhr100.csv")[:7]
    summary = []
    sheet_items = []
    for row in rows[:12]:
        sid = row["sample_id"]
        rgb = Image.open(row["image_path"]).convert("RGB")
        alpha = feather_alpha(read_alpha(amap[(sid, CURRENT)], rgb.size), 1.2)
        rgba = rgb.convert("RGBA")
        rgba.putalpha(Image.fromarray((alpha * 255 + 0.5).astype(np.uint8), mode="L"))
        sample_dir = out_root / "visuals" / sid
        sample_dir.mkdir(parents=True, exist_ok=True)
        rgba.save(sample_dir / "rgba.png")
        variants = {
            "original": rgb,
            "alpha_edge": edge_overlay(rgb, alpha),
            "replace_white": composite(rgb, alpha, make_bg(rgb.size, "white", rgb)),
            "replace_blue": composite(rgb, alpha, make_bg(rgb.size, "blue", rgb)),
            "replace_gradient": composite(rgb, alpha, make_bg(rgb.size, "gradient", rgb)),
            "blur_bg": composite(rgb, alpha, make_bg(rgb.size, "blur", rgb)),
        }
        for name, img in variants.items():
            img.save(sample_dir / f"{name}.jpg", quality=92)
        contact_sheet(list(variants.items()), sample_dir / "contact_sheet.jpg", cols=3, thumb=(280, 220))
        sheet_items.append((sid, Image.open(sample_dir / "contact_sheet.jpg")))
        summary.append({"sample_id": sid, "rgba": str(sample_dir / "rgba.png"), "contact_sheet": str(sample_dir / "contact_sheet.jpg")})
    contact_sheet(sheet_items, out_root / "visuals/overview_contact_sheet.jpg", cols=2, thumb=(520, 380))
    with (out_root / "ppt_assets/composer_assets.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "rgba", "contact_sheet"])
        w.writeheader()
        w.writerows(summary)
    report = """# PromptMatte Composer Showcase

- Base alpha: PromptMatte-TTA-GF (`zim_vitb_flip_tta_bbox_guided_r1`)
- Outputs: RGBA export, replace background, blur background, alpha-edge overlay.
- External basis: image matting evaluates alpha as a compositing primitive, not only a mask; promptable segmentation supplies the spatial object prior.
- Claim boundary: this is a presentation/application module, not a new metric SOTA claim.
"""
    (out_root / "FINAL_COMPOSER_SHOWCASE.md").write_text(report, encoding="utf-8")
    print(f"COMPOSER_DONE:{out_root}")


if __name__ == "__main__":
    main()
