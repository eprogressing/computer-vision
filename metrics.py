"""
metrics.py - 视频抠图评价指标
与队友图像分支的评价指标完全一致
基于 MatAnyone 官方评估脚本
"""

import torch
import numpy as np
import cv2
from typing import List, Dict, Union, Tuple
from pathlib import Path

# ============ 复用 MatAnyone 官方指标 ============
from evaluation.eval_yt_hr import MetricMAD, MetricMSE, MetricGRAD

# ============ 图像级指标（与队友图像分支一致）============

def compute_sad(alpha_pred: np.ndarray, alpha_gt: np.ndarray) -> float:
    """
    Sum of Absolute Differences (SAD)
    与队友图像分支使用的指标完全一致
    
    Args:
        alpha_pred: 预测 alpha，范围 [0,1] 或 [0,255]
        alpha_gt: Ground Truth alpha，范围 [0,1] 或 [0,255]
    """
    if alpha_pred.max() > 1:
        alpha_pred = alpha_pred / 255.0
    if alpha_gt.max() > 1:
        alpha_gt = alpha_gt / 255.0
    
    sad = np.sum(np.abs(alpha_pred - alpha_gt))
    return float(sad)


def compute_mse(alpha_pred: np.ndarray, alpha_gt: np.ndarray) -> float:
    """Mean Squared Error (MSE)"""
    if alpha_pred.max() > 1:
        alpha_pred = alpha_pred / 255.0
    if alpha_gt.max() > 1:
        alpha_gt = alpha_gt / 255.0
    
    mse = np.mean((alpha_pred - alpha_gt) ** 2)
    return float(mse)


def compute_mae_x1000(alpha_pred: np.ndarray, alpha_gt: np.ndarray) -> float:
    """MAE × 1000 - 与队友的 MAE_x1000 完全一致"""
    if alpha_pred.max() > 1:
        alpha_pred = alpha_pred / 255.0
    if alpha_gt.max() > 1:
        alpha_gt = alpha_gt / 255.0
    
    # 转换为 tensor 使用官方 MetricMAD
    pred_tensor = torch.from_numpy(alpha_pred).float()
    gt_tensor = torch.from_numpy(alpha_gt).float()
    
    mad = MetricMAD()
    return float(mad(pred_tensor, gt_tensor))


def compute_gradient_error(alpha_pred: np.ndarray, alpha_gt: np.ndarray) -> float:
    """Gradient Error - 使用 MatAnyone 官方实现"""
    if alpha_pred.max() > 1:
        alpha_pred = alpha_pred / 255.0
    if alpha_gt.max() > 1:
        alpha_gt = alpha_gt / 255.0
    
    pred_tensor = torch.from_numpy(alpha_pred).float()
    gt_tensor = torch.from_numpy(alpha_gt).float()
    
    grad = MetricGRAD()
    return float(grad(pred_tensor, gt_tensor))


def compute_boundary_sad(alpha_pred: np.ndarray, alpha_gt: np.ndarray,
                          boundary_width: int = 5) -> float:
    """Boundary SAD - 专注发丝等精细边缘"""
    if alpha_pred.max() > 1:
        alpha_pred = alpha_pred / 255.0
    if alpha_gt.max() > 1:
        alpha_gt = alpha_gt / 255.0
    
    # 找到 GT 的边界区域
    gt_binary = (alpha_gt > 0.5).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (boundary_width, boundary_width))
    dilated = cv2.dilate(gt_binary, kernel)
    eroded = cv2.erode(gt_binary, kernel)
    boundary_mask = dilated - eroded
    
    if boundary_mask.sum() == 0:
        return 0.0
    
    boundary_sad = np.sum(np.abs(alpha_pred - alpha_gt) * boundary_mask)
    return float(boundary_sad)


# ============ 视频级指标 ============

def compute_temporal_consistency(alpha_sequence: List[np.ndarray]) -> float:
    """时序一致性 - 帧间差异均值（无参考指标）"""
    if len(alpha_sequence) <= 1:
        return 0.0
    
    total_diff = 0.0
    for t in range(len(alpha_sequence) - 1):
        alpha_t = alpha_sequence[t]
        alpha_t1 = alpha_sequence[t + 1]
        if alpha_t.max() > 1:
            alpha_t = alpha_t / 255.0
        if alpha_t1.max() > 1:
            alpha_t1 = alpha_t1 / 255.0
        
        diff = np.mean(np.abs(alpha_t - alpha_t1))
        total_diff += diff
    
    return total_diff / (len(alpha_sequence) - 1)


def compute_dtssd(alpha_seq_pred: List[np.ndarray], 
                  alpha_seq_gt: List[np.ndarray]) -> float:
    """
    DTSSD (delta-t Sum of Squared Differences) - 有参考时序指标
    
    衡量预测的帧间变化模式是否与 GT 的帧间变化模式一致。
    与 RVM/MaGGIe/MatAnyone 官方评估脚本完全一致。
    
    公式: dtSSD = sqrt(mean([(pred_t - pred_{t-1}) - (true_t - true_{t-1})]^2)) * 100
    
    - 值越小越好（0 = 完美，预测的帧间变化与 GT 完全一致）
    - 与 TC（无参考）不同，DTSSD 利用 GT 区分"好的变化"和"坏的抖动"
    
    Args:
        alpha_seq_pred: 预测 alpha 序列，每帧范围 [0,1] 或 [0,255]
        alpha_seq_gt: GT alpha 序列，每帧范围 [0,1] 或 [0,255]
    
    Returns:
        DTSSD 值（已乘以 100 缩放）
    """
    assert len(alpha_seq_pred) == len(alpha_seq_gt), "序列长度不匹配"
    if len(alpha_seq_pred) <= 1:
        return 0.0
    
    total_dtssd = 0.0
    for t in range(1, len(alpha_seq_pred)):
        pred_t = alpha_seq_pred[t].astype(np.float32)
        pred_tm1 = alpha_seq_pred[t - 1].astype(np.float32)
        true_t = alpha_seq_gt[t].astype(np.float32)
        true_tm1 = alpha_seq_gt[t - 1].astype(np.float32)
        
        # 归一化到 [0,1]
        if pred_t.max() > 1:
            pred_t /= 255.0
        if pred_tm1.max() > 1:
            pred_tm1 /= 255.0
        if true_t.max() > 1:
            true_t /= 255.0
        if true_tm1.max() > 1:
            true_tm1 /= 255.0
        
        # dtSSD = sqrt(mean(((pred_t - pred_{t-1}) - (true_t - true_{t-1}))^2))
        dt_diff = (pred_t - pred_tm1) - (true_t - true_tm1)
        dtssd_frame = np.sqrt(np.mean(dt_diff ** 2))
        total_dtssd += dtssd_frame
    
    # 平均所有帧间对，乘以 100（与官方一致）
    return (total_dtssd / (len(alpha_seq_pred) - 1)) * 100.0


# ============ 综合评估（与队友格式一致）============

def evaluate_alpha(pred_alpha: np.ndarray, gt_alpha: np.ndarray) -> Dict[str, float]:
    """
    单帧评估，返回与队友图像分支完全一致的指标格式
    """
    return {
        'SAD': compute_sad(pred_alpha, gt_alpha),
        'MSE': compute_mse(pred_alpha, gt_alpha),
        'MAE_x1000': compute_mae_x1000(pred_alpha, gt_alpha),
        'Gradient': compute_gradient_error(pred_alpha, gt_alpha),
        'Boundary_SAD': compute_boundary_sad(pred_alpha, gt_alpha)
    }


def evaluate_video(alpha_seq_pred: List[np.ndarray], 
                    alpha_seq_gt: List[np.ndarray]) -> Dict[str, float]:
    """
    视频序列评估，返回所有指标的平均值
    """
    assert len(alpha_seq_pred) == len(alpha_seq_gt), "序列长度不匹配"
    
    n_frames = len(alpha_seq_pred)
    results = {}
    
    # 累加各帧指标
    sad_sum = mse_sum = mae_sum = grad_sum = boundary_sum = 0.0
    
    for t in range(n_frames):
        frame_results = evaluate_alpha(alpha_seq_pred[t], alpha_seq_gt[t])
        sad_sum += frame_results['SAD']
        mse_sum += frame_results['MSE']
        mae_sum += frame_results['MAE_x1000']
        grad_sum += frame_results['Gradient']
        boundary_sum += frame_results['Boundary_SAD']
    
    return {
        'SAD': sad_sum / n_frames,
        'MSE': mse_sum / n_frames,
        'MAE_x1000': mae_sum / n_frames,
        'Gradient': grad_sum / n_frames,
        'Boundary_SAD': boundary_sum / n_frames,
        'Temporal_Consistency': compute_temporal_consistency(alpha_seq_pred),
        'DTSSD': compute_dtssd(alpha_seq_pred, alpha_seq_gt)
    }


# ============ 测试 ============
if __name__ == "__main__":
    print("测试 metrics.py...")
    
    # 模拟数据
    h, w = 480, 864
    test_pred = np.random.rand(h, w).astype(np.float32)
    test_gt = np.random.rand(h, w).astype(np.float32)
    
    results = evaluate_alpha(test_pred, test_gt)
    print("\n单帧评估结果（与队友格式一致）:")
    for key, value in results.items():
        print(f"  {key}: {value:.6f}")
    
    print("\n metrics.py 创建成功！")