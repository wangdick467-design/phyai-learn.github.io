# 👁️ Vision Transformer (ViT) 学习与复现笔记

2021 年Transformer 架构成功从 NLP 领域跨界到计算机视觉（CV），而 **ViT (Vision Transformer)** 则是这一变革的里程碑之作。本篇笔记记录了 ViT 的核心设计思想、数学表达以及 PyTorch 核心代码实现。
---
## 🔗 论文基本信息与资源
* **论文题目**: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
* **发表会议**: ICLR 2021 (Oral)
* **论文原址**: [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
* **官方代码**: [google-research/vision_transformer](https://github.com/google-research/vision_transformer)


---

## 💡 核心思想：Img as 16x16 Words
传统的 CNN 依赖卷积核的局部感受野，而 ViT 完全抛弃了卷积操作，直接将图像分块（Patches）并转化为序列，送入标准的 Transformer Encoder。

其核心步骤可以概括为：
### 1. 图像分块 (Patch Embedding)
将输入图像 $X \in \mathbb{R}^{H \times W \times C}$ 均匀切分成不重叠的图像块 $X_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$。

其中 $(P,P)$ 是每个 Patch 的分辨率，而 Patch 的总个数（即序列长度）N 的计算公式为：
$$N = \frac{H \cdot W}{P^2}$$

### 2. 线性映射与位置编码
将每个 Patch 展平并通过线性层映射到维度 $D$，同时加入可学习的**位置编码 (Positional Embedding)** 以保留空间信息。

### 3. 引入 Class Token
参考 BERT，在输入序列最前方拼接一个可学习的 `[CLS]` token，其最终的输出状态将用于图像分类。

---

## 📐 数学表达 (Mathematical Formulations)

ViT 内部的前向传播公式如下：

### 阶梯一：输入序列初始化
（拼接 Class Token 并嵌入位置信息）
$$z_0 = [x_{\text{class}}; \, x_p^1 E; \, x_p^2 E; \, \dots; \, x_p^N E] + E_{\text{pos}}$$

其中：
* $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$ 是线性映射层的权重矩阵。
* $E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ 是可学习的位置编码。

### 阶梯二：Transformer Encoder 堆叠循环
对于层数 $l = 1, \dots, L$，每一层的计算流程为：
$$z_l' = \text{MSA}(\text{LN}(z_{l-1})) + z_{l-1}$$
$$z_l = \text{MLP}(\text{LN}(z_l')) + z_l'$$

> 📌 **注**：LN表示层归一化（Layer Normalization），MSA 表示多头自注意力机制（Multi-head Self-Attention）。

### 阶梯三：最终分类输出
$$y = \text{LN}(z_L^0)$$

只取最终输出 $$z_L$$ 的第 0个位置（即 `[CLS]` 对应的输出特征），送入 MLP Head 进行最终的类别预测。


## 🗺️ Vision Transformer (ViT) 模型框架总览

根据原论文的设计，ViT 模型的核心前向传播链条非常纯粹，主要由以下三个模块按顺序组成：
1. **Linear Projection of Flattened Patches (Embedding 层)**
2. **Transformer Encoder (多层编码器堆叠)**
3. **MLP Head (分类输出层)**

> 📌 **架构图占位**：*(在此处可以插入原论文的 Pipeline 架构图)*
> `![ViT Architecture]![ViT 架构图](./vit.png)`

---

## 🛠️ 第一部分：Embedding 层结构详解 (维度转换的艺术)

标准的 Transformer 模块专门为 NLP 设计，要求输入必须是二维矩阵序列：
$$\text{Shape: } [N, D] \quad (\text{即 } [\text{num\_token}, \, \text{token\_dim}])$$
以主流的 **ViT-B/16** 为例，每个 token 向量的长度硬性规定为 $D = 768$。

然而，图像数据的原生格式为三维矩阵：$[H, W, C]$。为了打通图像到序列的壁垒，Embedding 层执行了以下精妙的变换：

### 1. 图像分块与展平 (Patch Cutting)
* 以输入图片 $(224 \times 224 \times 3)$ 为例，若设定 Patch 大小为 $16 \times 16$，则一幅图会被切分成：
  $$N = \left(\frac{224}{16}\right)^2 = 14 \times 14 = 196 \text{ 个 Patches}$$
* 每一个 Patch 的原始数据维度为 $[16, 16, 3]$。将其拉伸展平后，得到一个长度为 $16 \times 16 \times 3 = 768$ 的一维向量。

### 2. 代码中的卷积工程实现
在实际代码复现中，直接进行全连接映射效率较低。工业界（包括官方源码）精妙地**通过一个二维卷积层**一步到位地实现了切块与投影：
* **核心参数**：`nn.Conv2d(in_channels=3, out_channels=768, kernel_size=16, stride=16)`
* **维度流向**：
  $$[224, \, 224, \, 3] \xrightarrow{\text{Conv2d}} [14, \, 14, \, 768] \xrightarrow{\text{Flatten HW}} [196, \, 768]$$
此时，数据完美转化为 Transformer 期望的二维矩阵。

### 3. 拼接 `[class]` token 与 Position Embedding
在送入 Encoder 之前，还需要完成两个学术层面的关键操作：
* **引入 Class Token**：借鉴 BERT 协议，在 196 个 tokens 最前方拼接一个可训练的特征向量 `[CLS]`（维度为 $[1, 768]$）。拼接后序列长度变为 $196 + 1 = 197$，维度变为 $[197, 768]$。
* **叠加位置编码 (Position Embedding)**：由于 Transformer 具有全向注意力（无视距离），必须加入一维可学习的位置参数（1D Pos. Emb.）来赋予模型空间几何感知。其维度同样为 $[197, 768]$，采用 **Add（直接矩阵相加）** 的方式融入。
  > 💡 **论文消融实验结论**：加入 1D 位置编码相比不使用，准确率大幅提升约 3%；而 1D 与 2D 位置编码在实际表现上几乎没有统计学差异。

---

## 🔄 第二部分：Transformer Encoder 详解

Transformer Encoder 的本质是 **Encoder Block 的 $L$ 次重复堆叠**（在 ViT-B 中 $L=12$）。单个 Block 内部由以下四个硬核组件闭环构成：



```text
输入 [197, 768] 
   │
   ├───> [ Layer Norm ] ──> [ Multi-Head Attention ] ──> [ Dropout/DropPath ] ──(+)──> 残差连接 1
   │                                                                            ▲
   └────────────────────────────────────────────────────────────────────────────┘
   │
   ├───> [ Layer Norm ] ──> [ MLP Block (节点翻4倍->缩回) ] ──> [ Dropout ] ───(+)──> 残差连接 2
   │                                                                            ▲
   └────────────────────────────────────────────────────────────────────────────┘
                                                                                │
                                                                       输出 [197, 768]


## 🎯 第三部分：MLP Head 详解与下游分类

经过 $L$ 层 Transformer Encoder 堆叠演化后，输出特征的 Shape 依然保持不变，仍为 $[197, 768]$。

1. **特征提取**：分类任务不需要完整的序列，因此我们**只抽取第 0 个位置的输出**（即由 `[class]` token 演变而来的最终特征），得到一个 $[1, 768]$ 的向量。
2. **Layer Norm 层**：在最终接入分类器前，原模型会再过一层论文中未画出的隐藏 **Layer Norm** 进行特征稳定。
3. **MLP Head 映射结构**：
   根据原论文设计，针对不同的任务阶段，MLP Head 的内部结构会有所不同：
   * **Pre-train 阶段（在大规模数据集如 ImageNet-21k 上预训练时）**：
     由 `Linear (全连接层) -> tanh (激活函数) -> Linear (全连接层)` 组成。它负责将 768 维的向量进一步映射，并通过非线性变换拟合海量的分类标签。
   * **Fine-tune 阶段（迁移到具体的下游分类任务时）**：
     为了保持结构的精简和高效，MLP Head 会退化为**单个 Linear 线性层**。它直接将特征从 768 维映射到你当前任务的类别数（如 $1000$ 类），最后通过 Softmax 输出最终的分类概率。

## 💻 PyTorch 核心代码实现 (Core Implementation)


以下是使用 PyTorch 实现 **Patch Embedding** 的核心片段，利用 `nn.Conv2d` 可以非常优雅地同时实现图像切块和线性映射：

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        
        # 利用步长(stride)等于 patch_size 的卷积巧妙实现切块与投影
        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x) # [B, embed_dim, H/P, W/P]
        x = x.flatten(2) # [B, embed_dim, N] (N = H/P * W/P)
        x = x.transpose(1, 2) # [B, N, embed_dim]
        return x

# 测试代码
if __name__ == "__main__":
    img = torch.randn(1, 3, 224, 224)
    patch_embed = PatchEmbedding()
    out = patch_embed(img)
    print(f"输入图像形状: {img.shape}")
    print(f"Patch Embedding 后序列形状: {out.shape} -> [Batch, Tokens, Embed_Dim]")
