"""
generate_composer_demo.py - 生成平滑后 alpha + 换背景的演示视频/GIF
支持自定义背景图片
"""

import cv2
import numpy as np
from pathlib import Path
import tempfile
from matanyone.inference.inference_core import InferenceCore
from matanyone.utils.get_default_model import get_matanyone_model
from matanyone.utils.device import get_default_device
from temporal_smooth import smooth_video_bidirectional
import os

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
    for ext in ['.png', '.jpg', '.jpeg']:
        candidate = pha_dir / f"0000{ext}"
        if candidate.exists():
            first_gt = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
            if first_gt is not None:
                break
    if first_gt is None:
        raise FileNotFoundError(f"First GT image not found in {pha_dir}")
    _, mask = cv2.threshold(first_gt, 128, 255, cv2.THRESH_BINARY)
    cv2.imwrite(output_mask_path, mask)

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

def load_background_image(bg_path, target_size):
    """加载背景图片并缩放到目标尺寸"""
    bg = cv2.imread(bg_path)
    if bg is None:
        raise FileNotFoundError(f"Background image not found: {bg_path}")
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
    bg = cv2.resize(bg, target_size)
    return bg

def create_composer_demo_video(rgb_frames, alphas, output_path, bg_image, fps=30):
    """
    创建换背景演示视频
    左：原始 RGB
    右：换背景效果（alpha 合成 + 自定义背景）
    """
    h, w = alphas[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w * 2, h))
    
    # 确保背景尺寸匹配
    if bg_image.shape[:2] != (h, w):
        bg_image = cv2.resize(bg_image, (w, h))
    
    for i in range(len(rgb_frames)):
        # 左侧：原始 RGB
        left = cv2.resize(rgb_frames[i], (w, h))
        left = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
        
        # 右侧：换背景效果
        alpha_normalized = alphas[i].astype(np.float32) / 255.0
        alpha_3ch = np.stack([alpha_normalized, alpha_normalized, alpha_normalized], axis=2)
        
        # 使用自定义背景
        right = (alpha_3ch * left + (1 - alpha_3ch) * bg_image).astype(np.uint8)
        
        # 左右拼接
        comparison = np.hstack([left, right])
        
        # 添加标签
        cv2.putText(comparison, "Original RGB", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(comparison, "Bidirectional + Custom BG", (w + 10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        out.write(comparison)
    
    out.release()
    print(f"Composer demo video saved to {output_path}")

def create_composer_demo_gif(rgb_frames, alphas, output_path, bg_image, duration=4, fps=30):
    """创建换背景演示 GIF"""
    import subprocess
    
    temp_video = output_path.replace('.gif', '_temp.mp4')
    create_composer_demo_video(rgb_frames, alphas, temp_video, bg_image, fps=15)
    
    cmd = [
        'ffmpeg', '-i', temp_video,
        '-vf', f'fps={fps},setpts=1.0*PTS,scale=720:-1,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
        '-t', str(duration),
        '-y', output_path
    ]
    subprocess.run(cmd, capture_output=True)
    
    os.remove(temp_video)
    print(f"Composer demo GIF saved to {output_path}")

def main():
    subset = "youtubematte_motion"
    scene_id = "0016"
    
    GT_BASE = "/mnt/d/学校作业/大三下/计算机视觉/GT/YouTubeMatte/youtubematte_512x288"
    scene_path = Path(GT_BASE) / subset / scene_id
    har_dir = scene_path / "har"
    pha_dir = scene_path / "pha"
    
    # ========== 在这里修改背景图片路径 ==========
    # 选项1：使用项目中的图片
    BG_IMAGE_PATH = "/home/user/MatAnyone/computer-vision/pic/background.jpg"  # 替换成你的背景图片路径
    
    # 选项2：使用队友的合成结果图片作为背景
    # BG_IMAGE_PATH = "pic/final/clean/04_0100_1_clean_alpha_composer.png"
    
    # 选项3：使用网络图片（先下载）
    # BG_IMAGE_PATH = "pic/beach.jpg"
    
    # 如果背景图片不存在，使用白色背景
    if not os.path.exists(BG_IMAGE_PATH):
        print(f"Warning: Background image not found at {BG_IMAGE_PATH}")
        print("Using white background instead...")
        bg_image = np.ones((1, 1, 3), dtype=np.uint8) * 255
    else:
        bg_image = cv2.imread(BG_IMAGE_PATH)
        bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
    
    print(f"Generating composer demo for {subset}/{scene_id}")
    
    # 加载模型
    print("Loading MatAnyone model...")
    device = get_default_device()
    model_path = "pretrained_models/matanyone.pth"
    model = get_matanyone_model(model_path)
    model = model.to(device).eval()
    processor = InferenceCore(model, cfg=model.cfg)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_video = os.path.join(tmpdir, "input.mp4")
        images_to_video(str(har_dir), input_video, fps=30)
        mask_path = os.path.join(tmpdir, "mask.png")
        generate_first_mask(pha_dir, mask_path)
        
        print("Running MatAnyone inference...")
        _, pred_alpha_video = processor.process_video(
            input_path=input_video,
            mask_path=mask_path,
            output_path=tmpdir,
            max_size=720,
            save_image=False
        )
        pred_alphas = read_alpha_from_video(pred_alpha_video)
        
        # 加载 RGB
        rgb_frames = load_rgb_sequence(har_dir, max_frames=len(pred_alphas))
        
        # 统一尺寸
        h_gt, w_gt = pred_alphas[0].shape
        rgb_frames_resized = []
        for rgb in rgb_frames:
            if rgb.shape[:2] != (h_gt, w_gt):
                rgb = cv2.resize(rgb, (w_gt, h_gt))
            rgb_frames_resized.append(rgb)
        
        # 调整背景尺寸
        if bg_image.shape[:2] != (h_gt, w_gt):
            bg_image = cv2.resize(bg_image, (w_gt, h_gt))
        
        # 应用 Bidirectional 平滑
        print("Applying Bidirectional smoothing (alpha=0.7)...")
        smoothed_alphas = smooth_video_bidirectional(pred_alphas, alpha=0.7)
        
        # 输出目录
        output_dir = Path("./demo_videos")
        output_dir.mkdir(exist_ok=True)
        
        # 1. 生成换背景演示视频
        composer_video = output_dir / f"{subset}_{scene_id}_bidirectional_composer.mp4"
        create_composer_demo_video(
            rgb_frames_resized, smoothed_alphas, 
            str(composer_video), bg_image
        )
        
        # 2. 生成换背景演示 GIF
        composer_gif = output_dir / f"{subset}_{scene_id}_bidirectional_composer.gif"
        create_composer_demo_gif(
            rgb_frames_resized, smoothed_alphas,
            str(composer_gif), bg_image,
            duration=3, fps=30
        )
        
        print(f"\nDemo files saved to {output_dir}/")
        print(f"  - {composer_video.name} (video)")
        print(f"  - {composer_gif.name} (GIF for slides)")

if __name__ == "__main__":
    main()