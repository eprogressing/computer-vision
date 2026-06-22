"""
video_runner.py - MatAnyone 视频推理封装
用于课程项目 PromptMatte 的视频抠图核心模块
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np
import cv2

# 添加 MatAnyone 路径
sys.path.insert(0, str(Path(__file__).parent))

from matanyone.inference.inference_core import InferenceCore
from matanyone.utils.get_default_model import get_matanyone_model
from matanyone.utils.device import get_default_device


class MatAnyoneVideoRunner:
    """
    MatAnyone 视频抠图运行器
    
    封装 MatAnyone 的推理流程，提供简单的 API
    支持单目标和多目标视频抠图
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: Optional[str] = None,
                 max_size: int = 480):
        """
        初始化 MatAnyone 运行器
        
        Args:
            model_path: 模型权重路径，默认使用预训练模型
            device: 运行设备，默认自动检测
            max_size: 输入视频最大边长（节省显存）
        """
        if device is None:
            device = get_default_device()
        self.device = device
        self.max_size = max_size
        
        # 加载模型
        if model_path is None:
            model_path = "pretrained_models/matanyone.pth"
        
        if not os.path.exists(model_path):
            # 尝试自动下载
            model_path = get_matanyone_model(model_path)
        
        self.model = get_matanyone_model(model_path)
        self.model = self.model.to(device).eval()
        
        # 创建推理处理器
        self.processor = InferenceCore(self.model, cfg=self.model.cfg)
    
    def process_video(self,
                      video_path: str,
                      first_frame_mask_path: str,
                      output_dir: Optional[str] = None,
                      suffix: str = "",
                      save_images: bool = False,
                      r_erode: int = 10,
                      r_dilate: int = 10,
                      n_warmup: int = 10) -> Tuple[str, str]:
        """
        处理视频，返回前景视频和 alpha 视频路径
        
        Args:
            video_path: 输入视频路径
            first_frame_mask_path: 第一帧掩膜路径（PNG，白色为前景）
            output_dir: 输出目录，默认使用临时目录
            suffix: 输出文件名后缀
            save_images: 是否保存单帧图片
            r_erode: 腐蚀半径
            r_dilate: 膨胀半径
            n_warmup: 预热帧数
        
        Returns:
            (foreground_video_path, alpha_video_path)
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 调用 MatAnyone 推理
        fgr_path, pha_path = self.processor.process_video(
            input_path=video_path,
            mask_path=first_frame_mask_path,
            output_path=output_dir,
            suffix=suffix,
            save_image=save_images,
            max_size=self.max_size,
            r_erode=r_erode,
            r_dilate=r_dilate,
            n_warmup=n_warmup
        )
        
        return fgr_path, pha_path
    
    def process_video_with_mask_array(self,
                                        video_frames: List[np.ndarray],
                                        first_frame_mask: np.ndarray,
                                        fps: int = 30,
                                        output_dir: Optional[str] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        使用 numpy 数组处理视频（用于 Gradio 界面）
        
        Args:
            video_frames: 视频帧列表 [RGB]
            first_frame_mask: 第一帧掩膜 [H, W]
            fps: 帧率
            output_dir: 输出目录
        
        Returns:
            (foreground_frames, alpha_frames)
        """
        # 保存临时文件
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        
        # 保存视频
        temp_video_path = os.path.join(output_dir, "temp_input.mp4")
        h, w = video_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (w, h))
        
        for frame in video_frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        writer.release()
        
        # 保存掩膜
        temp_mask_path = os.path.join(output_dir, "temp_mask.png")
        cv2.imwrite(temp_mask_path, (first_frame_mask * 255).astype(np.uint8))
        
        # 推理
        fgr_path, pha_path = self.process_video(
            temp_video_path, temp_mask_path, output_dir
        )
        
        # 读取结果
        foreground_frames = []
        alpha_frames = []
        
        cap_fgr = cv2.VideoCapture(fgr_path)
        cap_pha = cv2.VideoCapture(pha_path)
        
        while True:
            ret_fgr, frame_fgr = cap_fgr.read()
            ret_pha, frame_pha = cap_pha.read()
            
            if not ret_fgr or not ret_pha:
                break
            
            frame_fgr = cv2.cvtColor(frame_fgr, cv2.COLOR_BGR2RGB)
            frame_pha = cv2.cvtColor(frame_pha, cv2.COLOR_BGR2GRAY)
            
            foreground_frames.append(frame_fgr)
            alpha_frames.append(frame_pha)
        
        cap_fgr.release()
        cap_pha.release()
        
        return foreground_frames, alpha_frames
    
    def reset(self):
        """重置处理器状态"""
        self.processor.clear_memory()


if __name__ == "__main__":
    print("测试 video_runner.py...")
    
    # 测试初始化
    runner = MatAnyoneVideoRunner(max_size=480)
    print(f"设备: {runner.device}")
    print(f"最大尺寸: {runner.max_size}")
    print("模型加载成功！")
    
    # 测试推理（如果有测试文件）
    test_video = "inputs/video/test-sample1.mp4"
    test_mask = "inputs/mask/test-sample1.png"
    
    if os.path.exists(test_video) and os.path.exists(test_mask):
        print(f"\n测试推理: {test_video}")
        fgr, pha = runner.process_video(test_video, test_mask)
        print(f"前景输出: {fgr}")
        print(f"Alpha 输出: {pha}")
    else:
        print("\n跳过推理测试（测试文件不存在）")
    
    print("\n video_runner.py 创建成功！")