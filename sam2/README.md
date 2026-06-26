# SAM2 Pipeline — 上游粗分割推理与消融实验

> **项目**: PromptMatte — 基于视觉基础模型融合的文本驱动零训练精细抠图与换背景系统  
> **部分**: 上游目标定位（SAM2 基线 + 推理优化）

---

## 文件结构

```
sam2_submission/
├── README.md                              ← 本文件
├── config.py                              ← 配置文件（路径、方法定义）
├── sam2_inference.py                      ← 核心推理脚本（8 种方法）
├── compute_metrics.py                     ← 指标计算脚本（SAD/MSE/MAE/Boundary）
├── valid706_manifest_for_alignment.csv    ← valid706 样本清单
├── final_holdout506_manifest_for_alignment.csv ← holdout506 样本清单
└── outputs/                               ← 推理结果输出目录（运行后自动创建）
```

---

## 环境配置

### 1. Python 环境

```bash
python -m venv sam2_env
source sam2_env/bin/activate   # Linux
# 或 sam2_env\Scripts\activate  # Windows

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install opencv-python-headless scipy pillow numpy tqdm
```

### 2. 安装 SAM2

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
cd ..
```

### 3. 下载 SAM2 模型权重

```bash
mkdir checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_base_plus.pt
cd ..
```

### 4. 准备数据集

将 MicroMat-3K 数据集放在 `./MicroMat-3K/MicroMat3K/` 下，结构如下：

```
MicroMat-3K/
└── MicroMat3K/
    ├── img/
    │   ├── 0001.png
    │   ├── 0002.png
    │   └── ...
    └── gt/
        ├── 0001_1.png
        ├── 0001_2.png
        └── ...
```

### 5. 修改配置

编辑 `config.py` 中的路径：

```python
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_base_plus.pt"  # 模型权重路径
DATASET_BASE_OVERRIDE = Path("./MicroMat-3K")                  # 数据集根目录
```

---

## 运行方法

### 推理（生成 alpha mask）

```bash
# 运行单个方法
python sam2_inference.py --method sam2_ensemble_guided

# 运行所有 8 种方法（需在 config.py 的 METHODS 列表中定义）
# 不指定 --method 时默认运行 METHODS 列表中的全部
```

**8 种方法说明**:

| 方法名 | 说明 | 推理时间(706张) |
|--------|------|:--:|
| `sam2_bbox_binary` | 基线：bbox-only 单次推理 + 硬二值化 | ~1 min |
| `sam2_guided` | bbox + Guided Filter 边缘平滑 | ~2 min |
| `sam2_multiscale` | bbox + 3 尺度融合 (0.75×/1.0×/1.25×) | ~3 min |
| `sam2_multiscale_guided` | bbox + 多尺度 + Guided Filter | ~4 min |
| `sam2_ensemble_rerank` | bbox + pos points → 6 候选 IoU 重排序 | ~5 min |
| `sam2_ensemble_guided` | Ensemble Rerank + Guided Filter | ~6 min |
| `sam2_ensemble_multiscale` | Ensemble Rerank + 3 尺度融合 | ~10 min |
| `sam2_ensemble_multiscale_guided` | 全部叠加 (Ensemble + Multiscale + Guided) | ~12 min |

输出位置：`outputs/{方法名}/{sample_id}/alpha.png`

### 计算指标

```bash
# 计算指定方法的指标
python compute_metrics.py \
    --manifest valid706_manifest_for_alignment.csv \
    --predictions-dir outputs/ \
    --methods sam2_bbox_binary,sam2_ensemble_guided \
    --output-metrics-all outputs/metrics_all.csv \
    --output-leaderboard outputs/leaderboard.csv
```

输出：
- `metrics_all.csv`：每个样本 × 每个方法的详细指标
- `leaderboard.csv`：按方法汇总的平均指标

### 评测指标

| 指标 | 全称 | 含义 | 方向 |
|------|------|------|:--:|
| **SAD** | Sum of Absolute Differences | 预测 alpha 与 GT 的绝对误差总和 | ↓ |
| **MSE** | Mean Squared Error | 均方误差 | ↓ |
| **MAE×1000** | Mean Absolute Error × 1000 | 平均绝对误差（放大 1000 倍）| ↓ |
| **Boundary SAD** | Boundary SAD | 边界带（5px）内的 SAD | ↓ |

---

## 依赖

- Python ≥ 3.10
- PyTorch ≥ 2.0 (CUDA 推荐)
- SAM2 (facebookresearch/sam2)
- opencv-python-headless
- scipy
- numpy, pillow, tqdm
