# PromptMatte：基于一致性先验的零训练精细抠图与视频应用

计算机视觉课程大作业 · 第五组
梁景铭 · 史傅冠华 · 张竣翔　|　南开大学 计算机学院

## 简介

精细抠图需要在头发、毛发与半透明边界处估计连续透明度 α，是背景替换、虚化与视频合成的共同前提。
PromptMatte 是一个**零训练（training-free）**的精细抠图与换背景框架：不更新任何模型权重，
只复用现成的视觉基础模型，把全部设计放在推理阶段的**先验注入**上。

贯穿系统的是一条可跨域复用的**一致性先验**——正确的 α 在变换下应保持一致：图像端落地为
翻转等变的测试时增强与边缘投影（PromptMatte-TTA-GF），并辅以以检测框为空间先验的
Box-Support Prior；视频端沿时间轴推广为双向时序平滑。图像分支与视频分支因此不是彼此独立，
而是同一方法在空间与时间上的两次实例化。

## 流水线

- **图像主线**：图像 + official bbox → SAM2 候选掩码 → Prompt Ensemble + IoU 引导重排序 →
  引导滤波修边 → ZIM 输出连续 α → PromptMatte-TTA-GF + Box-Support Prior → RGBA / 换背景 / 背景虚化。
- **视频支线**：复用图像主线的首帧 α 作为种子 → MatAnyone2 逐帧推理 → 双向时序平滑稳定边界。

整体流水线图见 `report/figures/pipeline.png`。

## 仓库结构

| 路径 | 负责人 | 内容 |
|------|--------|------|
| `report/` | 全体 | 课程报告 LaTeX 源码与 `main.pdf`（CVPR 中文模板，正文 8 页 + 附录） |
| `sam2/` | 史傅冠华 | 上游定位：SAM2 推理（8 种方法）、Prompt Ensemble + IoU 重排序、指标计算 |
| `zim/` | 梁景铭 | 精细抠图：PromptMatte-TTA-GF + Box-Support Prior 实现、评测流水线与结果表 |
| `video/` | 张竣翔 | 视频时序平滑：MatAnyone2 推理封装、7 种平滑策略、批量实验与指标 |
| `第五组+梁景铭-史傅冠华-张竣翔.pptx` | 全体 | 答辩 PPT |

`sam2/` 与 `zim/` 各自附带独立 README 与运行说明。

## 主要结果

**图像：MicroMat-3K valid706（706 张，统一 official bbox 与评分协议；四项指标均越低越好）**

| 阶段 | 方法 | SAD | MSE | MAE×10³ | Bnd-SAD |
|------|------|----:|----:|-------:|-------:|
| 上游分割 | bbox 二值基线 | 3.314 | 0.000956 | 1.149 | 1.432 |
| | + ensemble + guided（最优） | 2.774 | 0.000747 | 0.981 | 1.216 |
| 精细抠图 | ZIM bbox 基线 | 2.320 | 0.000580 | 0.825 | 0.737 |
| | + PromptMatte-TTA-GF | 2.124 | 0.000446 | 0.753 | 0.639 |
| | **+ Box-Support Prior（最终）** | **2.123** | **0.000445** | **0.753** | **0.638** |

相对 ZIM 基线，最终方法在 SAD / MSE / MAE / Boundary-SAD 上分别下降 **8.5% / 23.1% / 8.8% / 13.4%**；
即便上游分割优化到最优（SAD 2.774）也不及 ZIM 基线（2.320），印证了精细抠图基座不可替代。

**视频：20 段（YouTubeMatte + VideoMatt 各 10）；TC 越高越好，SAD/MSE 变化越低越好**

| 策略 | TC 提升 | SAD 变化 | MSE 变化 | 细节 |
|------|-------:|------:|------:|:--:|
| Median | 6.68% | +8.88% | +23.91% | 差 |
| **Bidirectional（采用）** | **2.89%** | **+0.44%** | **−0.12%** | **优** |
| EMA | 2.59% | +1.64% | +2.40% | 良 |

双向平滑是唯一在几乎无精度损失下提升时序一致性的策略（完整 7 种策略对比见报告表 2）。

## 分工

- **梁景铭** — 精细抠图与系统整合：ZIM + PromptMatte-TTA-GF + Box-Support Prior（`zim/`）
- **史傅冠华** — 上游开放词汇定位：SAM2 + Prompt Ensemble + IoU 引导重排序（`sam2/`）
- **张竣翔** — 视频时序平滑：MatAnyone2 + 双向时序平滑（`video/`）

## 编译报告

需要 **XeLaTeX + ctex**（中文）：

```bash
cd report
latexmk -xelatex main.tex   # 产出 main.pdf
```

## 复现说明

各部分仅含源代码，不含模型权重、数据集与中间输出。运行所需环境、数据与命令见各子目录：

- 上游分割（SAM2）：见 `sam2/README.md`，需 SAM2.1 Hiera-Base+ 权重与 MicroMat-3K。
- 精细抠图（ZIM）：见 `zim/README.md`，基座为 ZIM ViT-B，引导滤波半径 1。
- 视频平滑：`video/`，依赖 MatAnyone2 与 YouTubeMatte / VideoMatt，双向平滑 α=0.7。
