"""
generate_paper_figures.py - 为论文报告生成图片和视频素材
使用测试视频，运行 MatAnyone 推理 + Bidirectional 平滑，
输出首帧对比图、换背景视频，并计算 DTSSD + Boundary 指标。
"""

import os
import cv2
import numpy as np
from pathlib import Path
import tempfile
import sys

# 直接使用底层组件
from matanyone.inference.inference_core import InferenceCore
from matanyone.utils.get_default_model import get_matanyone_model
from matanyone.utils.device import get_default_device
from temporal_smooth import smooth_video_bidirectional
from metrics import evaluate_video, compute_dtssd

# ============ 配置 ============
TEST_VIDEO = "/mnt/d/学校作业/大三下/计算机视觉/测试文件.mp4"
OUTPUT_DIR = Path("./paper_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# 平滑参数（论文推荐的最佳配置）
BIDIRECTIONAL_ALPHA = 0.7
# ============================

def read_alpha_from_video(video_path):
    """从预测的 alpha 视频读取序列"""
    cap = cv2.VideoCapture(video_path)
    alphas = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        alphas.append(frame)
    cap.release()
    return alphas

def read_rgb_from_video(video_path, max_frames=None):
    """从视频读取 RGB 帧"""
    cap = cv2.VideoCapture(video_path)
    rgbs = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and i >= max_frames:
            break
        rgbs.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return rgbs

def generate_first_mask_from_video(video_path, output_mask_path):
    """从视频首帧生成 mask（简单阈值）"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read video: {video_path}")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 使用 Otsu 阈值自动分割前景
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(output_mask_path, mask)
    print(f"Generated mask (Otsu): {output_mask_path}")
    return mask

def create_comparison_frame(rgb, alpha_orig, alpha_smooth, output_path):
    """
    创建首帧对比图：
    上排：原始 RGB | 原始 Alpha | 平滑 Alpha
    下排：原始换背景 | 平滑换背景
    """
    h, w = alpha_orig.shape
    
    # 统一 RGB 尺寸
    if rgb.shape[:2] != (h, w):
        rgb = cv2.resize(rgb, (w, h))
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    # Alpha 转 BGR 可视化
    alpha_orig_bgr = cv2.cvtColor(alpha_orig, cv2.COLOR_GRAY2BGR)
    alpha_smooth_bgr = cv2.cvtColor(alpha_smooth, cv2.COLOR_GRAY2BGR)
    
    # 换背景（使用绿色背景便于观察边缘）
    bg_color = (0, 177, 64)  # 绿色背景
    bg = np.full((h, w, 3), bg_color, dtype=np.uint8)
    
    # 原始 alpha 换背景
    alpha_norm_orig = alpha_orig.astype(np.float32) / 255.0
    alpha_3ch_orig = np.stack([alpha_norm_orig]*3, axis=2)
    comp_orig = (alpha_3ch_orig * rgb_bgr.astype(np.float32) + (1 - alpha_3ch_orig) * bg.astype(np.float32)).astype(np.uint8)
    
    # 平滑 alpha 换背景
    alpha_norm_smooth = alpha_smooth.astype(np.float32) / 255.0
    alpha_3ch_smooth = np.stack([alpha_norm_smooth]*3, axis=2)
    comp_smooth = (alpha_3ch_smooth * rgb_bgr.astype(np.float32) + (1 - alpha_3ch_smooth) * bg.astype(np.float32)).astype(np.uint8)
    
    # 拼接：上排 3 张，下排 2 张（居中）
    top_row = np.hstack([rgb_bgr, alpha_orig_bgr, alpha_smooth_bgr])
    
    # 下排两张居中
    spacer = np.full((h, w//2, 3), 255, dtype=np.uint8)
    bottom_row = np.hstack([comp_orig, spacer, comp_smooth])
    
    # 如果下排宽度不匹配，调整
    if bottom_row.shape[1] != top_row.shape[1]:
        bottom_row = cv2.resize(bottom_row, (top_row.shape[1], h))
    
    result = np.vstack([top_row, bottom_row])
    
    # 添加标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    labels_top = ["Original RGB", "Original Alpha", "Smoothed Alpha (Bidirectional)"]
    labels_bottom = ["Composite (Original)", "", "Composite (Smoothed)"]
    
    for i, label in enumerate(labels_top):
        x = i * w + 10
        cv2.putText(result, label, (x, 25), font, 0.5, (255, 255, 255), 2)
        cv2.putText(result, label, (x, 25), font, 0.5, (0, 0, 0), 1)
    
    for i, label in enumerate(labels_bottom):
        if label:
            x = i * (w//2) + 10
            cv2.putText(result, label, (x, h + 25), font, 0.5, (255, 255, 255), 2)
            cv2.putText(result, label, (x, h + 25), font, 0.5, (0, 0, 0), 1)
    
    cv2.imwrite(output_path, result)
    print(f"Comparison frame saved: {output_path}")

def create_composer_video(rgb_frames, alphas, output_path, bg_color=(0, 177, 64), fps=30):
    """创建换背景演示视频"""
    h, w = alphas[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w * 2, h))
    
    bg = np.full((h, w, 3), bg_color, dtype=np.uint8)
    
    for i in range(min(len(rgb_frames), len(alphas))):
        rgb = rgb_frames[i]
        if rgb.shape[:2] != (h, w):
            rgb = cv2.resize(rgb, (w, h))
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        alpha_norm = alphas[i].astype(np.float32) / 255.0
        alpha_3ch = np.stack([alpha_norm]*3, axis=2)
        comp = (alpha_3ch * rgb_bgr.astype(np.float32) + (1 - alpha_3ch) * bg.astype(np.float32)).astype(np.uint8)
        
        comparison = np.hstack([rgb_bgr, comp])
        cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(comparison, "Bidirectional Smoothed + Green BG", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        out.write(comparison)
    
    out.release()
    print(f"Composer video saved: {output_path}")

def main():
    print("=" * 70)
    print("Paper Figure Generation - MatAnyone + Bidirectional Smoothing")
    print("=" * 70)
    
    # 检查测试视频
    if not os.path.exists(TEST_VIDEO):
        print(f"ERROR: Test video not found: {TEST_VIDEO}")
        sys.exit(1)
    
    # 加载模型
    print("\n[1/5] Loading MatAnyone model...")
    device = get_default_device()
    model_path = "pretrained_models/matanyone.pth"
    model = get_matanyone_model(model_path)
    model = model.to(device).eval()
    processor = InferenceCore(model, cfg=model.cfg)
    print(f"  Model loaded on {device}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 生成首帧 mask
        print("\n[2/5] Generating first-frame mask...")
        mask_path = os.path.join(tmpdir, "mask.png")
        generate_first_mask_from_video(TEST_VIDEO, mask_path)
        
        # 运行 MatAnyone 推理
        print("\n[3/5] Running MatAnyone inference on test video...")
        _, pred_alpha_video = processor.process_video(
            input_path=TEST_VIDEO,
            mask_path=mask_path,
            output_path=tmpdir,
            max_size=720,
            save_image=False
        )
        pred_alphas = read_alpha_from_video(pred_alpha_video)
        print(f"  Predicted {len(pred_alphas)} alpha frames")
        
        # 读取原始 RGB
        rgb_frames = read_rgb_from_video(TEST_VIDEO, max_frames=len(pred_alphas))
        print(f"  Read {len(rgb_frames)} RGB frames")
        
        # 统一尺寸
        h_gt, w_gt = pred_alphas[0].shape
        rgb_frames_resized = []
        for rgb in rgb_frames:
            if rgb.shape[:2] != (h_gt, w_gt):
                rgb = cv2.resize(rgb, (w_gt, h_gt))
            rgb_frames_resized.append(rgb)
        
        # 应用 Bidirectional 平滑
        print(f"\n[4/5] Applying Bidirectional smoothing (alpha={BIDIRECTIONAL_ALPHA})...")
        smoothed_alphas = smooth_video_bidirectional(pred_alphas, alpha=BIDIRECTIONAL_ALPHA)
        
        # 计算指标（无 GT，仅展示平滑前后的自对比）
        # 使用 TC 和 DTSSD 自对比
        tc_orig = evaluate_video(pred_alphas, pred_alphas)  # 自对比 TC
        tc_smooth = evaluate_video(smoothed_alphas, smoothed_alphas)
        
        # 计算帧间变化
        def compute_self_dtssd(alphas):
            """计算自参考 DTSSD（帧间 alpha 变化量）"""
            if len(alphas) < 2:
                return 0.0
            diffs = []
            for t in range(1, len(alphas)):
                diff = alphas[t].astype(np.float32) - alphas[t-1].astype(np.float32)
                diffs.append(np.mean(diff ** 2))
            return float(np.sqrt(np.mean(diffs)) * 100)
        
        dtssd_orig = compute_self_dtssd(pred_alphas)
        dtssd_smooth = compute_self_dtssd(smoothed_alphas)
        
        print(f"\n  Self-reference metrics (no GT available):")
        print(f"  Original TC: {tc_orig.get('Temporal_Consistency', 0):.6f}")
        print(f"  Smoothed TC: {tc_smooth.get('Temporal_Consistency', 0):.6f}")
        print(f"  Original self-DTSSD: {dtssd_orig:.4f}")
        print(f"  Smoothed self-DTSSD: {dtssd_smooth:.4f}")
        print(f"  DTSSD reduction: {(1 - dtssd_smooth/dtssd_orig)*100:.2f}%")
        
        # 生成首帧对比图
        print(f"\n[5/5] Generating output files...")
        comparison_path = OUTPUT_DIR / "first_frame_comparison.png"
        create_comparison_frame(
            rgb_frames_resized[0], 
            pred_alphas[0], 
            smoothed_alphas[0], 
            str(comparison_path)
        )
        
        # 保存首帧原始 alpha 和平滑 alpha（单独保存，便于论文引用）
        cv2.imwrite(str(OUTPUT_DIR / "first_frame_alpha_original.png"), pred_alphas[0])
        cv2.imwrite(str(OUTPUT_DIR / "first_frame_alpha_smoothed.png"), smoothed_alphas[0])
        cv2.imwrite(str(OUTPUT_DIR / "first_frame_rgb.png"), 
                    cv2.cvtColor(rgb_frames_resized[0], cv2.COLOR_RGB2BGR))
        
        # 生成换背景视频
        composer_path = OUTPUT_DIR / "test_video_bidirectional_composer.mp4"
        create_composer_video(rgb_frames_resized, smoothed_alphas, str(composer_path))
        
        # 保存平滑后的 alpha 视频
        alpha_video_path = OUTPUT_DIR / "test_video_smoothed_alpha.mp4"
        h, w = smoothed_alphas[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(alpha_video_path), fourcc, 30, (w, h), isColor=False)
        for a in smoothed_alphas:
            out.write(a)
        out.release()
        print(f"Smoothed alpha video saved: {alpha_video_path}")
        
        # 保存指标到文件
        metrics_path = OUTPUT_DIR / "metrics.txt"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write("MatAnyone + Bidirectional Smoothing Metrics\n")
            f.write("=" * 50 + "\n")
            f.write(f"Test video: {TEST_VIDEO}\n")
            f.write(f"Smoothing: Bidirectional (alpha={BIDIRECTIONAL_ALPHA})\n")
            f.write(f"Frames: {len(pred_alphas)}\n")
            f.write(f"Resolution: {w_gt}x{h_gt}\n\n")
            f.write(f"Self-reference TC (original): {tc_orig.get('Temporal_Consistency', 0):.6f}\n")
            f.write(f"Self-reference TC (smoothed): {tc_smooth.get('Temporal_Consistency', 0):.6f}\n")
            f.write(f"Self-DTSSD (original): {dtssd_orig:.4f}\n")
            f.write(f"Self-DTSSD (smoothed): {dtssd_smooth:.4f}\n")
            f.write(f"DTSSD reduction: {(1 - dtssd_smooth/dtssd_orig)*100:.2f}%\n")
        
        print(f"\n{'='*70}")
        print(f"All outputs saved to: {OUTPUT_DIR}/")
        print(f"  - first_frame_comparison.png  (6-panel comparison)")
        print(f"  - first_frame_rgb.png")
        print(f"  - first_frame_alpha_original.png")
        print(f"  - first_frame_alpha_smoothed.png")
        print(f"  - test_video_bidirectional_composer.mp4")
        print(f"  - test_video_smoothed_alpha.mp4")
        print(f"  - metrics.txt")
        print(f"{'='*70}")

if __name__ == "__main__":
    main()
