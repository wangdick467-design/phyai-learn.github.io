# Transformer 笔记

## 1. 核心背景

- **提出论文**: *Attention Is All You Need* (Vaswani et al., 2017)
网址：https://arxiv.org/pdf/1706.03762
- **主要动机**: 解决 RNN 的串行计算瓶颈和长距离依赖问题
- **核心创新**: 完全基于 **自注意力机制**，摒弃循环与卷积结构
> 吴恩达：https://www.bilibili.com/video/BV1JnV36vEeB/?spm_id_from=333.337.search-card.all.click
> 可视化：https://www.bilibili.com/video/BV13z421U7cs/?spm_id_from=333.337.search-card.all.click
---
https://www.bilibili.com/video/BV1ih4y1J7rx?spm_id_from=333.788.recommend_more_video.1&trackid=web_related_0.router-related-2589621-z4htq.1781257732974.238&vd_source=bd38266a7bcac2690321bce7d1e25954

Transformer 是由 Vaswani 等人在 2017 年提出的深度学习模型，其论文为《Attention Is All You Need》。

其核心思想：

> 使用 Self-Attention（自注意力）机制代替传统 RNN/LSTM 的序列建模方式，实现并行计算，提高长距离依赖建模能力。

目前广泛应用于：

- 自然语言处理（NLP）
- 计算机视觉（ViT）
- 多模态模型（LLaVA、Qwen-VL）
- 大语言模型（GPT、LLaMA、Qwen）

---

# 2. Transformer整体结构

Transformer 由两部分组成：

```text
           Input
             │
      ┌────────────┐
      │  Encoder   │ × N
      └────────────┘
             │
      ┌────────────┐
      │  Decoder   │ × N
      └────────────┘
             │
          Output
```

原始论文中：

- Encoder：6层
- Decoder：6层

---

# 3. Encoder结构

单个 Encoder Block：

```text
Input
  │
  ▼
Multi-Head Attention
  │
Add & Norm
  │
Feed Forward
  │
Add & Norm
  ▼
Output
```
<img width="626" height="331" alt="image" src="https://github.com/user-attachments/assets/abad2a14-f5c5-44ce-b41b-5e8afe0b8f55" />

#包含两大模块：

1. Multi-Head Self Attention
2. Feed Forward Network

多头注意里用于计算输入的注意力权重生成一个带有编码信息的输出向量 指示序列中每个词如何关注其他词
将多头注意力输出向量加到原始输入叫残差连接 残差连接的输出经过层归一化 残差连接有助于梯度直接流过网络
---

# 4. 输入表示（Input Embedding）

文本：

```text
"I love AI"
```

经过 Tokenizer：

```text
[101, 2057, 999]
```

Embedding 后：

```text
x₁, x₂, x₃
```

维度：

```text
d_model = 512
```

即：

```math
X \in \mathbb{R}^{n \times d_{model}}
```

其中：

- n：序列长度
- d_model：特征维度

---

# 5. Positional Encoding

Transformer 本身不具有顺序信息，因此需要加入位置编码：

```math
Input = Embedding + PositionEncoding
```

## 正弦位置编码

偶数维：

```math
PE(pos,2i)
=
\sin
\left(
\frac{pos}
{10000^{2i/d_{model}}}
\right)
```

奇数维：

```math
PE(pos,2i+1)
=
\cos
\left(
\frac{pos}
{10000^{2i/d_{model}}}
\right)
```

特点：

- 无需学习参数
- 可以外推到更长序列
- 具有相对位置信息

---

# 6. Self-Attention机制

Transformer 最核心的模块。

## Step1：生成 Q、K、V

输入：

```math
X
```

线性映射：

```math
Q = XW_Q
```

```math
K = XW_K
```

```math
V = XW_V
```

维度：

```math
Q,K,V \in \mathbb{R}^{n \times d_k}
```

---

## Step2：计算注意力分数

```math
Score = QK^T
```

得到：

```math
n \times n
```

矩阵。

---

## Step3：缩放

```math
\frac{QK^T}{\sqrt{d_k}}
```

原因：

防止内积过大导致 Softmax 梯度过小。

---

## Step4：Softmax归一化

```math
A=
Softmax
\left(
\frac{QK^T}
{\sqrt{d_k}}
\right)
```

得到注意力权重矩阵。

---

## Step5：加权求和

Transformer 核心公式：

```math
Attention(Q,K,V)
=
Softmax
\left(
\frac{QK^T}
{\sqrt{d_k}}
\right)V
```

最终输出：

```math
Z = AV
```

---

# 7. Self-Attention示例

句子：

```text
I love AI
```

假设计算单词：

```text
AI
```

对应 Query。

与：

```text
I
love
AI
```

分别计算相关性：

```text
0.1
0.3
0.6
```

Softmax 后：

```text
10%
30%
60%
```

说明：

```text
AI 更关注自身信息
```

从而形成上下文表示。

---

# 8. Multi-Head Attention

单头注意力：

```math
head_1
=
Attention(Q,K,V)
```

Transformer 采用多头机制：

```math
head_i
=
Attention(Q_i,K_i,V_i)
```

多个头并行：

```math
MultiHead(Q,K,V)
=
Concat(head_1,...,head_h)W_O
```

原论文参数：

```text
h = 8
```

---

## 多头注意力的意义

不同头学习不同关系：

```text
Head1 → 语法关系

Head2 → 主谓关系

Head3 → 长距离依赖

Head4 → 语义关系
```

从而提升模型表达能力。

---

# 9. Add & Norm

残差连接：

```math
Y = X + Attention(X)
```

随后进行 Layer Normalization：

```math
Output = LayerNorm(Y)
```

作用：

- 防止梯度消失
- 加快收敛
- 提高训练稳定性

---

# 10. Feed Forward Network

每个 Token 独立通过两层 MLP：

```math
FFN(x)
=
W_2
(ReLU(W_1x+b_1))
+b_2
```

结构：

```text
512
 ↓
2048
 ↓
512
```

特点：

- 不同 Token 参数共享
- 增强非线性表达能力

---

# 11. Decoder结构

Decoder 比 Encoder 多一个模块：

```text
Masked Multi-Head Attention
          ↓
Encoder-Decoder Attention
          ↓
Feed Forward
```

---
<img width="307" height="303" alt="image" src="https://github.com/user-attachments/assets/c3a5dc20-61f6-4974-b1ba-b0e1a03d9412" />

## Masked Attention

训练时：

```text
我 爱 人工智能
```

预测：

```text
人工智能
```

不能看到未来 Token。

Mask矩阵：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

实现：

```text
自回归生成
```

---

## Encoder-Decoder Attention

Query 来自 Decoder：

```math
Q = Decoder
```

Key 和 Value 来自 Encoder：

```math
K,V = Encoder
```

用于：

- 机器翻译
- 文本摘要
- 问答系统

---

# 12. Transformer训练流程

```text
Input Sentence
       │
Tokenizer
       │
Embedding
       │
Position Encoding
       │
Encoder
       │
Decoder
       │
Linear
       │
Softmax
       │
Prediction
```

损失函数：

```math
CrossEntropyLoss
```

---

# 13. Transformer优缺点

## 优点

### 并行计算

RNN：

```text
x1 → x2 → x3 → x4
```

Transformer：

```text
x1
x2
x3
x4
同时计算
```

训练速度显著提高。

---

### 长距离依赖能力强

例如：

```text
The animal didn't cross the street
because it was too tired.
```

Transformer 可以直接建立：

```text
it → animal
```

之间的联系。

---

## 缺点

### 计算复杂度高

Attention矩阵大小：

```math
n \times n
```

复杂度：

```math
O(n^2)
```

---

### 显存占用大

随着上下文长度增长：

```text
8K
16K
32K
128K
```

Attention 开销急剧增加。

---

# 14. Transformer发展路线

```text
Transformer (2017)
        │
        ├── BERT
        │
        ├── GPT
        │
        ├── T5
        │
        ├── ViT
        │
        ├── CLIP
        │
        ├── BLIP
        │
        ├── LLaVA
        │
        ├── Qwen-VL
        │
        └── GPT-4o
```

---

# 15. 核心公式汇总

## Query、Key、Value

```math
Q=XW_Q
```

```math
K=XW_K
```

```math
V=XW_V
```

---

## Scaled Dot Product Attention

```math
Attention(Q,K,V)
=
Softmax
\left(
\frac{QK^T}
{\sqrt{d_k}}
\right)V
```

---

## Multi-Head Attention

```math
MultiHead(Q,K,V)
=
Concat(head_1,...,head_h)W_O
```

---

## Feed Forward

```math
FFN(x)
=
W_2(ReLU(W_1x+b_1))+b_2
```

---

# 16. Transformer 与 GPT/BERT 的关系

## BERT

特点：

- Encoder Only
- 双向注意力
- 适合理解任务

例如：

- 文本分类
- 命名实体识别
- 情感分析

---

## GPT

特点：

- Decoder Only
- Masked Self-Attention
- 自回归生成

例如：

- 对话生成
- 代码生成
- 大语言模型

---

# 17. 一句话总结

> Transformer 的核心是利用 Self-Attention 建立序列中任意位置之间的关系，通过 Multi-Head Attention 学习不同层次特征，并利用并行计算实现高效训练，是现代大语言模型和多模态模型的基础架构。



> 一句话总结：Transformer = 全局注意力 + 位置编码 + FFN + 残差与归一化，通过堆叠层数实现强大的序列建模能力。
