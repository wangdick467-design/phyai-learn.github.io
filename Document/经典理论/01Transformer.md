# Transformer 笔记

## 1. 核心背景

- **提出论文**: *Attention Is All You Need* (Vaswani et al., 2017)
网址：https://arxiv.org/pdf/1706.03762
- **主要动机**: 解决 RNN 的串行计算瓶颈和长距离依赖问题
- **核心创新**: 完全基于 **自注意力机制**，摒弃循环与卷积结构
> 吴恩达：https://www.bilibili.com/video/BV1JnV36vEeB/?spm_id_from=333.337.search-card.all.click
> 可视化：https://www.bilibili.com/video/BV13z421U7cs/?spm_id_from=333.337.search-card.all.click
---

## 2. 整体架构

> Transformer 是一个 **Encoder-Decoder** 结构


### 2.1 Encoder（编码器）
- 由 **N=6** 个相同层堆叠（原论文）
- 每层包含两个子层：
  1. **Multi-Head Self-Attention**（多头自注意力）
  2. **Feed-Forward Network**（前馈网络，FFN）
- 每个子层后都跟随 **残差连接 + Layer Normalization**

### 2.2 Decoder（解码器）
- 同样由 **N=6** 个相同层堆叠
- 每层包含三个子层：
  1. **Masked Multi-Head Self-Attention**（掩码自注意力，防止看到未来位置）
  2. **Cross-Attention**（与编码器输出做注意力）
  3. **Feed-Forward Network**
- 每个子层后同样有 **残差 + Layer Norm**

---

## 3. 核心组件详解

### 3.1 Self-Attention（缩放点积注意力）

**输入**: Query (Q), Key (K), Value (V)

**公式**：
\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
\]

**步骤**：
1. 计算 Q 与 K 的相似度 → 得分矩阵
2. 缩放（除以 \(\sqrt{d_k}\)）防止梯度消失
3. Softmax 归一化 → 注意力权重
4. 加权求和 V → 输出

### 3.2 Multi-Head Attention（多头注意力）

- **思想**: 将 Q, K, V 线性投影到多个不同的低维空间（h=8 头），并行计算注意力，最后拼接再线性变换。
- **作用**: 让模型从不同子空间（不同“角度”）学习信息。

\[
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O
\]
\[
\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
\]

### 3.3 Positional Encoding（位置编码）

原因：Transformer 本身没有时序/位置概念。

**使用正弦/余弦函数**（固定编码）：
\[
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\]
\[
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\]

- 每个位置 \(pos\) 得到唯一的 \(d_{\text{model}}\) 维向量
- 与词嵌入直接相加后输入模型

### 3.4 Feed-Forward Network（FFN）

每个位置独立使用同一个两层全连接 + ReLU：

\[
\text{FFN}(x) = \max(0, xW_1 + b_1) W_2 + b_2
\]

- 典型维度：\(d_{\text{model}}=512\)，内层 \(d_{ff}=2048\)

### 3.5 Layer Normalization & Residual Connection

- **残差**: \(x + \text{Sublayer}(x)\)，缓解梯度消失
- **LayerNorm**: 对特征维度归一化，加速收敛

---

## 4. 关键技巧

| 技术 | 位置 | 作用 |
|------|------|------|
| Mask（掩码） | Decoder 的自注意力 | 屏蔽当前位置之后的 token，保证自回归 |
| 共享权重 | 各层独立 | 一般各层参数不共享（除非特意设计） |
| Dropout | 各子层之后 | 正则化 |

---

## 5. 训练细节（原论文配置）

| 参数 | 值 |
|------|----|
| 层数 N | 6 |
| 模型维度 \(d_{\text{model}}\) | 512 |
| FFN 内层维度 \(d_{ff}\) | 2048 |
| 注意力头数 h | 8 |
| 每个头的维度 \(d_k = d_v\) | \(512/8 = 64\) |
| Dropout | 0.1 |
| 优化器 | Adam (betas=0.9, 0.98) |
| 学习率 | 动态调整：\(\text{rate} = d_{\text{model}}^{-0.5} \cdot \min(\text{step}^{-0.5}, \text{step} \cdot \text{warmup}^{-1.5})\)，warmup=4000 步 |
| Batch Size | 约 25000 词（动态 batching） |

---

## 6. 优点与局限

### ✅ 优点
- 全局建模能力强（直接计算任意两位置关系）
- 可并行计算（训练速度快）
- 容易堆叠深层

### ⚠️ 局限
- 计算复杂度 \(O(n^2 \cdot d)\)，长序列内存开销大
- 缺少天然的局部归纳偏置（需要大数据或预训练）
- 位置编码是固定的，无法自适应

---

## 7. 常见改进与变体

- **BERT**: 仅使用 Encoder，双向注意力，预训练+微调
- **GPT**: 仅使用 Decoder，因果自回归生成
- **T5**: 原始的 Encoder-Decoder，统一 NLP 任务为文本转换
- **Transformer-XL / XLNet**: 处理超长序列，片段递归
- **Longformer / BigBird**: 降低复杂度到 \(O(n\log n)\) 或 \(O(n)\)
- **相对位置编码**（如 RoPE, ALiBi）

---

## 8. 手写理解检查清单

- [ ] 为什么需要缩放因子 \(\sqrt{d_k}\)？
- [ ] 残差连接 + LayerNorm 的顺序区别（Post-Norm vs Pre-Norm）？
- [ ] Decoder 的交叉注意力中，Q 从哪来？K, V 从哪来？
- [ ] 如何保证自回归生成时不看到未来 token？
- [ ] 多头注意力中每个头能否学习不同模式？
- [ ] 位置编码和词嵌入的维度为什么要一致？

---

> 一句话总结：Transformer = 全局注意力 + 位置编码 + FFN + 残差与归一化，通过堆叠层数实现强大的序列建模能力。
