# 👁️ Vision Transformer (ViT) 学习与复现笔记

近年来，Transformer 架构成功从 NLP 领域跨界到计算机视觉（CV），而 **ViT (Vision Transformer)** 则是这一变革的里程碑之作。本篇笔记记录了 ViT 的核心设计思想、数学表达以及 PyTorch 核心代码实现。

---

## 💡 核心思想：Img as 16x16 Words
传统的 CNN 依赖卷积核的局部感受野，而 ViT 完全抛弃了卷积操作，直接将图像分块（Patches）并转化为序列，送入标准的 Transformer Encoder。

其核心步骤可以概括为：
1. **图像分块 (Patch Embedding)**：将输入图像 $X \in \mathbb{R}^{H \times W \times C}$ 均匀切分成不重叠的图像块 $X_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$，其中 $(P, P)$ 是每个 Patch 的分辨率，$N = HW/P^2$ 是 Patch 的总个数（即序列长度）。
2. **线性映射与位置编码**：将每个 Patch 展平并通过线性层映射到维度 $D$，同时加入可学习的**位置编码 (Positional Embedding)** 以保留空间信息。
3. **引入 Class Token**：参考 BERT，在输入序列最前方拼接一个可学习的 `[CLS]` token，其最终的输出状态将用于图像分类。

---

## 📐 数学表达 (Mathematical Formulations)

ViT 内部的前向传播公式如下：

1. **输入序列初始化**（拼接 Class Token 并嵌入位置信息）：
   $$z_0 = [x_{\text{class}}; \, x_p^1 E; \, x_p^2 E; \, \dots; \, x_p^N E] + E_{\text{pos}}$$
   其中 $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$ 是线性映射层，$E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ 是位置编码。

2. **Transformer Encoder 堆叠循环**（对于层数 $l = 1, \dots, L$）：
   $$z_l' = \text{MSA}(\text{LN}(z_{l-1})) + z_{l-1}$$
   $$z_l = \text{MLP}(\text{LN}(z_l')) + z_l'$$
   *注：$\text{LN}$ 表示 Layer Normalization，$\text{MSA}$ 表示 Multi-head Self-Attention。*

3. **最终分类输出**：
   $$y = \text{LN}(z_L^0)$$
   只取 $z_L$ 的第 0 个位置（即 `[CLS]` 对应的输出）送入 MLP Head 进行最终预测。

---

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
