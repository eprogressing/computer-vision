# PromptMatte — 计算机视觉课程大作业（第五组）

> 基于视觉基础模型融合的**零训练（training-free）**精细抠图与换背景系统
> 作者：梁景铭、史傅冠华、张竣翔 · 南开大学计算机学院

## 方法概览

**图像主线：** 图像 + official bbox → SAM2 候选掩码 → Prompt Ensemble + IoU 重排序 →
Guided Filter 修边 → ZIM 软 alpha → PromptMatte-TTA-GF + Box-Support Prior →
RGBA / 换背景 / 背景虚化。

**视频支线：** 复用图像主线首帧 alpha → MatAnyone2 → 双向时序平滑。

核心是一条贯穿空间与时间的「一致性先验」：图像端为翻转等变（TTA-GF），
视频端为时序一致（双向时序平滑），全程不训练任何模型权重。

## 仓库结构

| 路径 | 内容 |
|------|------|
| `report/` | CVPR 中文模板课程报告（`main.tex`，XeLaTeX；正文 8 页 + 附录），产物见 `report/main.pdf` |
| `第五组+梁景铭-史傅冠华-张竣翔.pptx` | 答辩 PPT |
| `code/`（待上传） | 三部分源代码：`sam/`（史傅冠华）、`zim/`（梁景铭）、`video/`（张竣翔） |

## 分工

- **梁景铭**：精细抠图与系统整合（ZIM + PromptMatte-TTA-GF + Box-Support Prior）
- **史傅冠华**：上游开放词汇分割（SAM2 + Prompt Ensemble + IoU 引导重排序）
- **张竣翔**：视频时序平滑（MatAnyone2 + 双向时序平滑）

## 编译报告

需要 **XeLaTeX + ctex**（中文）。

```bash
cd report
latexmk -xelatex main.tex      # 产出 main.pdf
```
