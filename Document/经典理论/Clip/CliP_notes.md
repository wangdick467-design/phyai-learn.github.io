# CLIP 学习笔记（Contrastive Language-Image Pretraining）

> 作者：OpenAI  
> 论文：Learning Transferable Visual Models From Natural Language Supervision（2021）  
> 模型名称：CLIP（Contrastive Language-Image Pretraining）

> https://github.com/openai/CLIP

> 论文连接https://arxiv.org/abs/2103.00020

> https://www.bilibili.com/video/BV1SL4y1s7LQ?spm_id_from=333.788.videopod.sections&vd_source=9629687338410a5ccaa5e1a595d0f17d
---

# 1. CLIP简介

CLIP（Contrastive Language-Image Pretraining）是 OpenAI 在 2021 年提出的视觉-语言预训练模型（Vision-Language Model, VLM）。

核心思想：

> 利用互联网海量图文对（Image-Text Pair），通过对比学习（Contrastive Learning）将图像和文本映射到同一个特征空间。

传统图像分类：

```text
Image
  ↓
CNN
  ↓
Label
```

例如：

```text
猫
狗
汽车
飞机
```

必须人工标注类别。

---

CLIP：

```text
Image          Text
  │              │
Image Encoder  Text Encoder
  │              │
   └────Embedding Space────┘
                │
          Similarity
```

无需固定类别。

---

# 2. CLIP解决了什么问题

传统监督学习：

```text
ImageNet
```

训练：

```text
图片 → 类别
```

例如：

```text
图片 → Cat
```

存在问题：

- 类别有限
- 泛化能力差
- 标注成本高

---

CLIP：

训练数据：

```text
图片 + 网络文本描述
```

例如：

```text
图片：一只猫

文本：
"a photo of a cat"
```

互联网天然存在大量此类数据。

因此：

```text
无限类别
```

成为可能。

---

# 3. 核心思想

CLIP的目标：

```text
正确图文
距离近

错误图文
距离远
```

例如：

```text
猫图片
```

对应：

```text
"a cat"
```

距离近。

---

对应：

```text
"a car"
```

距离远。

---

训练目标：

```text
Image ↔ Text Alignment
```

即：

让图像和文本理解统一到同一个语义空间。

---

# 4. CLIP整体架构

结构：

```text
          Image
             │
             ▼
      Image Encoder
             │
             ▼
       Image Feature
             │
             ▼

         Similarity

             ▲
             │
       Text Feature
             ▲
             │
       Text Encoder
             ▲
             │
            Text
```

---

# 5. 图像编码器

CLIP支持两种视觉骨干网络：

---

## CNN版本

ResNet：

```text
ResNet50

ResNet101

ResNet50x4

ResNet50x16
```

---

结构：

```text
Image
 ↓
ResNet
 ↓
Feature
 ↓
Projection
 ↓
Embedding
```

---

## ViT版本

Vision Transformer

```text
ViT-B/32

ViT-B/16

ViT-L/14
```

---

结构：

```text
Image
 ↓
Patch
 ↓
Transformer
 ↓
CLS Token
 ↓
Projection
 ↓
Embedding
```

---

实际应用中：

```text
ViT效果更好
```

---

# 6. 文本编码器

CLIP使用Transformer Encoder。

输入：

```text
"a photo of a cat"
```

Tokenizer：

```text
[a, photo, of, a, cat]
```

---

结构：

```text
Text
 ↓
Tokenizer
 ↓
Embedding
 ↓
Transformer
 ↓
Feature
 ↓
Projection
 ↓
Embedding
```

---

输出：

```math
T \in R^d
```

---

# 7. 特征投影层

图像特征：

```math
f_i
```

文本特征：

```math
f_t
```

维度可能不同。

因此增加：

```math
W_i
```

```math
W_t
```

投影矩阵。

---

得到：

```math
z_i
```

```math
z_t
```

统一到：

```math
R^{512}
```

空间。

---

# 8. 特征归一化

CLIP使用：

```math
L2 Normalize
```

即：

```math
z
=
\frac{z}
{||z||}
```

---

作用：

只保留方向信息。

方便计算：

```math
Cosine Similarity
```

---

# 9. 相似度计算

图像：

```math
z_i
```

文本：

```math
z_t
```

相似度：

```math
S
=
z_i^T z_t
```

由于归一化：

```math
S
=
Cosine Similarity
```

---

范围：

```text
-1 ~ 1
```

---

# 10. 对比学习（Contrastive Learning）

CLIP最核心部分。

---

假设一个Batch：

```text
Image1 ↔ Text1

Image2 ↔ Text2

Image3 ↔ Text3
```

---

构建相似度矩阵：

```text
        T1  T2  T3

I1      ✓

I2          ✓

I3              ✓
```

---

目标：

对角线最大。

非对角线最小。

---

即：

```text
正确配对
距离近

错误配对
距离远
```

---

# 11. InfoNCE损失

CLIP采用：

```text
Contrastive Loss
```

具体：

```math
L_i
=
-\log
\frac
{
exp(s_{ii}/\tau)
}
{
\sum_j exp(s_{ij}/\tau)
}
```

---

文本方向同理：

```math
L_t
=
-\log
\frac
{
exp(s_{ii}/\tau)
}
{
\sum_j exp(s_{ji}/\tau)
}
```

---

最终：

```math
L
=
\frac{L_i+L_t}{2}
```

---

# 12. CLIP训练流程

步骤：

```text
Image
 ↓
Image Encoder
 ↓
Image Embedding
```

同时：

```text
Text
 ↓
Text Encoder
 ↓
Text Embedding
```

---

计算：

```math
Similarity Matrix
```

---

然后：

```math
Contrastive Loss
```

---

最后：

```text
Backpropagation
```

更新参数。

---

# 13. Zero-Shot分类

CLIP最著名能力。

---

传统分类：

```text
训练：
猫
狗
汽车
```

只能识别：

```text
猫
狗
汽车
```

---

CLIP：

构造Prompt：

```text
"a photo of a cat"

"a photo of a dog"

"a photo of a car"
```

---

输入图片：

```text
未知图片
```

---

计算：

```math
Similarity
```

---

选择最大：

```math
argmax
```

即预测类别。

---

无需重新训练。

---

# 14. Zero-Shot推理流程

```text
Image
  │
  ▼
Image Encoder
  │
  ▼
Image Feature
  │

Compare

  │
  ▼

Text Features
(a cat)
(a dog)
(a car)
```

---

取：

```text
最高相似度
```

即可。

---

# 15. CLIP为什么成功

原因1：

```text
海量数据
```

训练集：

```text
4亿图文对
```

---

原因2：

```text
自然语言监督
```

不依赖人工类别。

---

原因3：

```text
对比学习
```

学习语义关系。

---

原因4：

```text
Transformer时代
```

视觉和语言统一建模。

---

# 16. CLIP与CNN分类器区别

| 项目 | CNN分类器 | CLIP |
|--------|--------|--------|
| 标签 | 固定类别 | 自然语言 |
| 泛化 | 差 | 强 |
| 新类别 | 重新训练 | 直接推理 |
| 监督方式 | 人工标注 | 图文对 |
| Zero-Shot | 不支持 | 支持 |

---

# 17. CLIP的局限性

---

## 细粒度识别较弱

例如：

```text
狗品种
```

容易混淆。

---

## OCR能力有限

文字理解较弱。

---

## 推理能力弱

CLIP：

```text
看图
```

擅长。

---

但：

```text
复杂推理
```

较差。

---

# 18. CLIP对后续模型影响

CLIP开创：

```text
Vision + Language
```

统一表征路线。

---

后续模型：

```text
ALIGN
BLIP
BLIP2
Flamingo
LLaVA
MiniGPT4
Qwen-VL
GPT-4V
```

几乎全部受到CLIP影响。

---

# 19. CLIP与LLaVA关系

LLaVA结构：

```text
Image
 ↓
CLIP ViT
 ↓
Projector
 ↓
LLM
 ↓
Answer
```

其中：

```text
CLIP ViT
```

负责：

```text
视觉特征提取
```

---

因此：

> LLaVA本质上是在CLIP视觉编码器基础上接入大语言模型。

---

# 20. 核心公式总结

图像编码：

```math
I \rightarrow z_i
```

文本编码：

```math
T \rightarrow z_t
```

---

归一化：

```math
z=\frac{z}{||z||}
```

---

相似度：

```math
S=z_i^Tz_t
```

---

InfoNCE损失：

```math
L
=
-\log
\frac
{
exp(s_{ii}/\tau)
}
{
\sum_j exp(s_{ij}/\tau)
}
```

---

# 21. 面试高频问题

### Q1：CLIP为什么能Zero-Shot？

因为训练目标不是固定类别分类，而是学习图像与文本的语义对齐。

---

### Q2：CLIP为什么使用对比学习？

让正确图文距离近，错误图文距离远。

---

### Q3：CLIP中的文本编码器是什么？

Transformer Encoder。

---

### Q4：CLIP中的视觉编码器是什么？

ResNet 或 ViT。

---

### Q5：LLaVA为什么使用CLIP？

CLIP已经学习到高质量视觉语义空间，可直接作为视觉编码器。

---

# 一句话总结

> CLIP通过图像编码器和文本编码器分别提取特征，并利用对比学习将图像和文本映射到统一语义空间，使模型具备强大的跨模态理解和Zero-Shot泛化能力，是现代视觉语言模型（LLaVA、Qwen-VL、GPT-4V等）的重要基础。
