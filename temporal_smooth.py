"""
temporal_smooth.py - 时序平滑模块（支持 EMA、高斯、中值、自适应、引导EMA、光流、双向）
"""

import numpy as np
import cv2
from typing import List, Optional

class TemporalSmoother:
    def __init__(self, window_size: int = 3, sigma: float = 1.5,
                 alpha: float = 0.9, mode: str = 'ema',
                 diff_threshold: float = 0.08,
                 flow_sigma: float = 1.0):
        self.window_size = window_size
        self.sigma = sigma
        self.alpha = alpha
        self.mode = mode
        self.diff_threshold = diff_threshold
        self.flow_sigma = flow_sigma  # 用于光流模式
        self.buffer = []          # 存储 alpha 历史
        self.rgb_buffer = []      # 存储 RGB 历史（用于 guided_ema 和 flow）

    def reset(self):
        self.buffer = []
        self.rgb_buffer = []

    def smooth_frame(self, current_alpha: np.ndarray, current_rgb: Optional[np.ndarray] = None) -> np.ndarray:
        # 归一化
        if current_alpha.max() > 1:
            current_alpha = current_alpha.astype(np.float32) / 255.0
        self.buffer.append(current_alpha.copy())
        if current_rgb is not None:
            self.rgb_buffer.append(current_rgb.copy())
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
            if self.rgb_buffer:
                self.rgb_buffer.pop(0)

        # 窗口不足时直接返回原图
        if len(self.buffer) < max(3, self.window_size // 2 + 1):
            return (current_alpha * 255).astype(np.uint8)

        if self.mode == 'ema':
            if len(self.buffer) >= 2:
                result = self.alpha * current_alpha + (1 - self.alpha) * self.buffer[-2]
            else:
                result = current_alpha
        elif self.mode == 'gaussian':
            T = len(self.buffer)
            weights = np.exp(-np.square(np.arange(T) - (T-1)) / (2 * self.sigma**2))
            weights = weights / weights.sum()
            stacked = np.stack(self.buffer, axis=0)
            result = np.zeros_like(current_alpha)
            for i in range(T):
                result += weights[i] * stacked[i]
        elif self.mode == 'median':
            T = len(self.buffer)
            stacked = np.stack(self.buffer, axis=0)
            result = np.median(stacked, axis=0)
        elif self.mode == 'adaptive':
            if len(self.buffer) >= 2:
                prev = self.buffer[-2]
                diff = np.abs(current_alpha - prev)
                flicker_mask = diff > self.diff_threshold
                boundary_mask = (current_alpha > 0.1) & (current_alpha < 0.9)
                apply_mask = flicker_mask & boundary_mask
                result = current_alpha.copy()
                if np.any(apply_mask):
                    result[apply_mask] = self.alpha * current_alpha[apply_mask] + \
                                         (1 - self.alpha) * prev[apply_mask]
            else:
                result = current_alpha
        elif self.mode == 'guided_ema':
            if len(self.buffer) >= 2 and current_rgb is not None and self.rgb_buffer:
                prev_alpha = self.buffer[-2]
                prev_rgb = self.rgb_buffer[-2]
                curr_rgb = current_rgb
                diff = np.abs(current_alpha - prev_alpha)
                flicker_mask = diff > self.diff_threshold
                boundary_mask = (current_alpha > 0.1) & (current_alpha < 0.9)
                apply_mask = flicker_mask & boundary_mask
                if not np.any(apply_mask):
                    result = current_alpha
                else:
                    guide = curr_rgb.astype(np.float32) / 255.0
                    src = current_alpha
                    guided = cv2.ximgproc.guidedFilter(guide=guide, src=src, radius=4, eps=0.01)
                    result = current_alpha.copy()
                    result[apply_mask] = self.alpha * guided[apply_mask] + \
                                         (1 - self.alpha) * prev_alpha[apply_mask]
            else:
                result = current_alpha
        elif self.mode == 'flow':
            # 光流运动补偿平滑
            if len(self.buffer) >= 2 and current_rgb is not None and self.rgb_buffer:
                prev_alpha = self.buffer[-2]
                prev_rgb = self.rgb_buffer[-2]
                curr_rgb = current_rgb
                # 转换为灰度图计算光流
                prev_gray = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
                curr_gray = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY)
                # 计算稠密光流（Farneback）
                flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                # 扭曲前一帧 alpha
                h, w = current_alpha.shape
                map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
                map_x = (map_x + flow[..., 0]).astype(np.float32)
                map_y = (map_y + flow[..., 1]).astype(np.float32)
                warped_prev = cv2.remap(prev_alpha, map_x, map_y, cv2.INTER_LINEAR)
                # 计算光流幅度，作为运动权重：运动越大，越依赖当前帧
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                # 权重：运动小时，更多依赖扭曲后的前一帧；运动大时，更多依赖当前帧
                # 使用 self.flow_sigma 控制衰减速度
                weight_prev = np.exp(- magnitude / self.flow_sigma)
                # 归一化：权重和为1
                weight_prev = np.clip(weight_prev, 0.1, 0.9)
                weight_curr = 1 - weight_prev
                # 加权平均
                result = weight_curr * current_alpha + weight_prev * warped_prev
                # 保护高置信区域
                core_mask = (current_alpha > 0.9) | (current_alpha < 0.1)
                result[core_mask] = current_alpha[core_mask]
            else:
                result = current_alpha
        else:
            result = current_alpha

        # 核心区域保护（所有模式通用）
        core_mask = (current_alpha > 0.9) | (current_alpha < 0.1)
        result[core_mask] = current_alpha[core_mask]
        result = np.clip(result, 0, 1)
        return (result * 255).astype(np.uint8)


def smooth_video_bidirectional(alpha_sequence: List[np.ndarray], alpha: float = 0.9) -> List[np.ndarray]:
    fwd_smoother = TemporalSmoother(mode='ema', alpha=alpha)
    fwd = []
    for a in alpha_sequence:
        fwd.append(fwd_smoother.smooth_frame(a))
    rev_smoother = TemporalSmoother(mode='ema', alpha=alpha)
    rev = []
    for a in reversed(alpha_sequence):
        rev.append(rev_smoother.smooth_frame(a))
    rev = list(reversed(rev))
    
    result = []
    for f, r in zip(fwd, rev):
        f_float = f.astype(np.float32)
        r_float = r.astype(np.float32)
        avg = (f_float + r_float) / 2
        result.append(avg.astype(np.uint8))
    return result