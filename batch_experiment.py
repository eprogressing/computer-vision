"""
batch_experiment_top10.py - 在 Top 10 抖动视频上批量测试所有平滑策略（包括双向平均和光流法），
输出每个视频的指标变化百分比，并汇总关键结果。
"""

import os
import cv2
import numpy as np
from pathlib import Path
import tempfile
import pandas as pd
from itertools import product

# 直接使用底层组件，避免重复初始化
from matanyone.inference.inference_core import InferenceCore
from matanyone.utils.get_default_model import get_matanyone_model
from matanyone.utils.device import get_default_device
from temporal_smooth import TemporalSmoother, smooth_video_bidirectional
from metrics import evaluate_video

# ============ 配置 ============
# YouTubeMatte
# TOP10_SCENES = [
#     ("youtubematte_static", "0005"),
#     ("youtubematte_motion", "0005"),
#     ("youtubematte_motion", "0026"),
#     ("youtubematte_static", "0026"),
#     ("youtubematte_motion", "0016"),
#     ("youtubematte_motion", "0007"),
#     ("youtubematte_static", "0007"),
#     ("youtubematte_static", "0016"),
#     ("youtubematte_motion", "0008"),
#     ("youtubematte_static", "0008"),
# ]
# GT_BASE = "/mnt/d/学校作业/大三下/计算机视觉/GT/YouTubeMatte/youtubematte_512x288"

# VideoMatt
TOP10_SCENES = [
    ("motion_set", "0040"),
    ("static_set", "0045"),
    ("static_set", "0030"),
    ("static_set", "0060"),
    ("motion_set", "0030"),
    ("static_set", "0075"),
    ("static_set", "0145"),
    ("static_set", "0025"),
    ("motion_set", "0025"),
    ("static_set", "0125"),
]
GT_BASE = "/mnt/d/学校作业/大三下/计算机视觉/GT/VideoMatt"
# ============ 定义参数网格（增加 flow 模式）============
PARAM_GRID = {
    'ema': {
        'alpha': [0.7, 0.9],
        'window_size': [3, 5, 7]
    },
    'gaussian': {
        'sigma': [1.5, 2.0],
        'window_size': [3, 5, 7]
    },
    'median': {
        'window_size': [3, 5, 7]
    },
    'adaptive': {
        'alpha': [0.5, 0.7, 0.9],
        'diff_threshold': [0.05, 0.08, 0.10],
        'window_size': [3]  # 占位
    },
    'guided_ema': {
        'alpha': [0.5, 0.7, 0.9],
        'diff_threshold': [0.05, 0.08, 0.10],
        'window_size': [3]
    },
    'bidirectional': {
        'alpha': [0.7, 0.9],
    },
    'flow': {   # 新增光流模式，只测试两个 alpha 值，窗口固定 3
        'alpha': [0.7, 0.9],
        'window_size': [3]   # 光流模式内部不使用窗口大小，占位
    }
}
# =================================

def images_to_video(image_dir, output_video, fps=30):
    images = sorted(Path(image_dir).glob("*.*"))
    if not images:
        raise FileNotFoundError(f"No images in {image_dir}")
    first = cv2.imread(str(images[0]))
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        out.write(img)
    out.release()
def generate_first_mask(pha_dir, output_mask_path):
    first_gt = None
    for start in ['0000', '0001']:  # 先尝试 0000，再尝试 0001
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = pha_dir / f"{start}{ext}"
            if candidate.exists():
                first_gt = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
                if first_gt is not None:
                    break
        if first_gt is not None:
            break
    if first_gt is None:
        raise FileNotFoundError(f"First GT image not found in {pha_dir}")
    _, mask = cv2.threshold(first_gt, 128, 255, cv2.THRESH_BINARY)
    cv2.imwrite(output_mask_path, mask)

def read_alpha_from_folder(pha_dir, max_frames=None):
    images = sorted(pha_dir.glob("*.png")) + sorted(pha_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No alpha images in {pha_dir}")
    alphas = []
    for i, img_path in enumerate(images):
        if max_frames is not None and i >= max_frames:
            break
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        alphas.append(img)
    return alphas

def read_alpha_from_video(video_path):
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

def load_rgb_sequence(har_dir, max_frames=None):
    images = sorted(har_dir.glob("*.*"))
    if max_frames:
        images = images[:max_frames]
    rgbs = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgbs.append(img)
    return rgbs
def evaluate_scene(subset, scene_id, processor):
    scene_path = Path(GT_BASE) / subset / scene_id
    har_dir = scene_path / "com"  # VideoMatt 用 com 目录（YouTubeMatte 用 har）
    pha_dir = scene_path / "pha"
    print(f"\n{'='*80}\nEvaluating {subset}/{scene_id}\n{'='*80}")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_video = os.path.join(tmpdir, "input.mp4")
        images_to_video(str(har_dir), input_video, fps=30)
        mask_path = os.path.join(tmpdir, "mask.png")
        generate_first_mask(pha_dir, mask_path)

        print("Running MatAnyone inference (once)...")
        _, pred_alpha_video = processor.process_video(
            input_path=input_video,
            mask_path=mask_path,
            output_path=tmpdir,
            max_size=720,
            save_image=False
        )
        pred_alphas = read_alpha_from_video(pred_alpha_video)
        gt_alphas = read_alpha_from_folder(pha_dir)

        min_len = min(len(pred_alphas), len(gt_alphas))
        pred_alphas = pred_alphas[:min_len]
        gt_alphas = gt_alphas[:min_len]

        h_gt, w_gt = gt_alphas[0].shape
        pred_alphas = [cv2.resize(p, (w_gt, h_gt), interpolation=cv2.INTER_CUBIC) for p in pred_alphas]

        rgb_frames = load_rgb_sequence(har_dir, max_frames=min_len)
        rgb_frames_resized = []
        for rgb in rgb_frames:
            if rgb.shape[:2] != (h_gt, w_gt):
                rgb = cv2.resize(rgb, (w_gt, h_gt))
            rgb_frames_resized.append(rgb)

        original_metrics = evaluate_video(pred_alphas, gt_alphas)
        print(f"Original metrics: SAD={original_metrics['SAD']:.4f}, TC={original_metrics['Temporal_Consistency']:.6f}, DTSSD={original_metrics['DTSSD']:.4f}\n")

        results = []
        
        # 打印表头
        print("-" * 160)
        print(f"{'Mode':<15} {'Params':<40} {'TC_change':>12} {'DTSSD_change':>14} {'SAD_change':>12} {'MSE_change':>12} {'Gradient':>12} {'Boundary':>12}")
        print("-" * 160)

        # 遍历所有模式
        for mode, params_dict in PARAM_GRID.items():
            if mode == 'bidirectional':
                for alpha in params_dict['alpha']:
                    smoothed = smooth_video_bidirectional(pred_alphas, alpha=alpha)
                    sm_metrics = evaluate_video(smoothed, gt_alphas)
                    changes = {}
                    for key in ['SAD', 'MSE', 'MAE_x1000', 'Gradient', 'Boundary_SAD', 'Temporal_Consistency', 'DTSSD']:
                        orig = original_metrics[key]
                        sm = sm_metrics[key]
                        changes[key] = ((sm - orig) / orig) * 100 if orig != 0 else 0
                    row = {'mode': mode, 'alpha': alpha}
                    row.update(changes)
                    results.append(row)
                    
                    # 表格输出
                    params_str = f"alpha={alpha}"
                    print(f"{mode:<15} {params_str:<40} {changes['Temporal_Consistency']:>+11.2f}% {changes['DTSSD']:>+13.2f}% {changes['SAD']:>+11.2f}% {changes['MSE']:>+11.2f}% {changes['Gradient']:>+11.2f}% {changes['Boundary_SAD']:>+11.2f}%")
                continue

            keys = list(params_dict.keys())
            values = list(params_dict.values())
            for combo in product(*values):
                params = dict(zip(keys, combo))
                if mode == 'ema':
                    smoother = TemporalSmoother(mode='ema', alpha=params['alpha'], window_size=params['window_size'])
                elif mode == 'gaussian':
                    smoother = TemporalSmoother(mode='gaussian', sigma=params['sigma'], window_size=params['window_size'])
                elif mode == 'median':
                    smoother = TemporalSmoother(mode='median', window_size=params['window_size'])
                elif mode == 'adaptive':
                    smoother = TemporalSmoother(mode='adaptive', alpha=params['alpha'],
                                                diff_threshold=params['diff_threshold'])
                elif mode == 'guided_ema':
                    smoother = TemporalSmoother(mode='guided_ema', alpha=params['alpha'],
                                                diff_threshold=params['diff_threshold'])
                elif mode == 'flow':
                    smoother = TemporalSmoother(mode='flow', alpha=params['alpha'])
                else:
                    continue

                smoother.reset()
                smoothed = []
                for i, alpha in enumerate(pred_alphas):
                    if mode in ['guided_ema', 'flow'] and i < len(rgb_frames_resized):
                        rgb = rgb_frames_resized[i]
                        smoothed.append(smoother.smooth_frame(alpha, rgb))
                    else:
                        smoothed.append(smoother.smooth_frame(alpha))

                sm_metrics = evaluate_video(smoothed, gt_alphas)
                changes = {}
                for key in ['SAD', 'MSE', 'MAE_x1000', 'Gradient', 'Boundary_SAD', 'Temporal_Consistency', 'DTSSD']:
                    orig = original_metrics[key]
                    sm = sm_metrics[key]
                    changes[key] = ((sm - orig) / orig) * 100 if orig != 0 else 0
                row = {'mode': mode}
                row.update(params)
                row.update(changes)
                results.append(row)
                
                # 表格输出
                params_str = ', '.join([f"{k}={v}" for k, v in params.items()])
                print(f"{mode:<15} {params_str:<40} {changes['Temporal_Consistency']:>+11.2f}% {changes['DTSSD']:>+13.2f}% {changes['SAD']:>+11.2f}% {changes['MSE']:>+11.2f}% {changes['Gradient']:>+11.2f}% {changes['Boundary_SAD']:>+11.2f}%")

        print("-" * 160)
        
        df = pd.DataFrame(results)
        csv_path = f"batch_results_{subset}_{scene_id}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nDetailed results saved to {csv_path}")
        return df, original_metrics
    
def main():
    print("Loading MatAnyone model (once)...")
    device = get_default_device()
    model_path = "pretrained_models/matanyone.pth"
    model = get_matanyone_model(model_path)
    model = model.to(device).eval()
    processor = InferenceCore(model, cfg=model.cfg)
    print("Model ready.\n")

    all_summaries = []
    for subset, scene_id in TOP10_SCENES:
        df, orig_metrics = evaluate_scene(subset, scene_id, processor)
        # 提取双向平滑 alpha=0.9 的结果（也可以提取光流 alpha=0.9 等）
        best_row = df[(df['mode'] == 'bidirectional') & (df['alpha'] == 0.9)]
        if not best_row.empty:
            row = best_row.iloc[0].to_dict()
            summary = {
                'scene': f"{subset}/{scene_id}",
                'orig_TC': orig_metrics['Temporal_Consistency'],
                'orig_DTSSD': orig_metrics['DTSSD'],
                'SAD_change': row.get('SAD', 0),
                'MSE_change': row.get('MSE', 0),
                'MAE_x1000_change': row.get('MAE_x1000', 0),
                'Gradient_change': row.get('Gradient', 0),
                'Boundary_SAD_change': row.get('Boundary_SAD', 0),
                'TC_change': row.get('Temporal_Consistency', 0),
                'DTSSD_change': row.get('DTSSD', 0)
            }
            all_summaries.append(summary)

    # 汇总表格
    summary_df = pd.DataFrame(all_summaries)
    summary_df = summary_df.round(2)
    print("\n" + "="*120)
    print("Top 10 视频上双向平滑 (alpha=0.9) 的指标变化汇总")
    print("="*120)
    print(summary_df.to_string(index=False))
    summary_df.to_csv("top10_bidirectional_summary.csv", index=False)
    print("\n汇总结果已保存到 top10_bidirectional_summary.csv")

if __name__ == "__main__":
    main()