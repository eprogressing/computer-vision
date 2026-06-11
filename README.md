# PromptMatte — 计算机视觉大作业幻灯片

> 基于视觉基础模型融合的**文本驱动零训练精细抠图与换背景**系统
> 作者：梁景铭、史傅冠华、张竣翔

## 模块化结构

| 文件 | 内容 | 负责人 |
|------|------|--------|
| `slides.tex` | 主文件：导言区（宏/配色/字体）+ 引言 + 精细抠图与实验 + 参考资料 | 梁景铭 |
| `sections/sam.tex` | 上游开放词汇分割（SAM / SAM2 / SAM3 / Grounded-SAM-2、候选选择等） | 史傅冠华 |
| `sections/video.tex` | 视频 matting 与应用（MatAnyone2、时序平滑、demo） | 张竣翔 |

主文件通过 `\input{sections/sam.tex}` 与 `\input{sections/video.tex}` **自动集成**两节，
无需改动主文件结构。

## 队友怎么编辑

只编辑自己的 `sections/*.tex`，在里面增删 `\begin{frame}[t]{标题} ... \end{frame}` 即可。
导言区的统一样式可直接复用（详见各 section 文件顶部注释、参考正文第 6–15 页）：

- `\hilite{重点}`、`\tcard{accent}{高度}{标题}{正文}`、`\ccard{accent}{高度}{居中正文}`
- `\notestrip{accent}{正文}`、`\metricchip{accent}{标签}{数值}`、`\mpill[accent]{标签}{数值}`、`\pill[accent]{文本}`
- 配色 `accent`：`nankai` / `accentblue` / `softgreen` / `softorange` / `softgray` / `deepred`

## 编译

需要 **XeLaTeX + ctex**（中文楷体）。

```bash
latexmk -xelatex slides.tex     # 推荐
# 或
xelatex slides.tex
```

## 协作流程

```bash
git clone https://github.com/eprogressing/computer-vision.git
cd computer-vision
# 编辑自己的 sections/xxx.tex
git add sections/xxx.tex
git commit -m "feat(sam): 添加 Grounded-SAM-2 流程页"
git push
```
