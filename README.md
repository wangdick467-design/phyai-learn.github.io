# phyai-learn.github.io
Share ai learning💡
---
## 展示自己多模态大模型学习的demo。物理学本，top2电子学博士Ai时代0基础自学Ai。

🗓️# 多模态大模型学习路线

## 阶段一：基础

### 1. 底层知识

#### 编程
- **Python** + **NumPy** / **Pandas**，能处理图像、文本、音频数据

#### 数学
- **线性代数**：向量、矩阵
- **概率统计**：分布、方差、期望、方差、贝叶斯定理
- **微积分**：梯度、导数、链式法则

#### 深度学习框架
- **PyTorch**（推荐）或 TensorFlow

### 2. 单模态基础

#### 视觉
- CNN原理
- ResNet
- 图像分类 / 检测
- 📖 参考：Stanford CS231n

#### 文本
- Transformer架构
- 自注意力机制
- 📖 参考：Stanford CS224N

---

## 阶段二：经典模型学习

### 

| 模型 | 全称 | 核心思想 |
|------|------|----------|
| **ViT** | Vision Transformer | 将图像划分为Patch，用Transformer替代CNN |
| **CLIP** | Contrastive Language-Image Pre-training | 图文对比预训练，对齐视觉与文本表征 |
| **BLIP** | Bootstrapping Language-Image Pre-training | 引导式语言图像预训练，提升图文理解 |
| **LXMERT** | Learning Cross-Modal Encoder Representations from Transformers | 学习跨模态编码器表示 |

## 阶段三：大型多模态语言模型部署-实战

#### LLaVA
- **L**arge **L**anguage-and-**V**ision **A**ssistant
- 视觉指令调优 (Visual Instruction Tuning)
#### GLM-4.1V-9B-Thinking
#### QwenVL系列

## 阶段四 VLA和世界模型
- VLA (左脑 - 决策与交互)： 传统 VLM（如 GPT-4V）只能看图说话。VLA 在此基础上加入了 Action（动作输出）。它把机械臂的控制指令（如七自由度关节角、末端位移）直接作为文本或 Token 预测出来。核心逻辑：把控制问题转化为序列生成问题。

- World Models (右脑 - 想象与预测)： 让智能体在“脑海”中模拟世界。输入当前状态和动作，预测未来的状态和奖励。核心逻辑：用自监督学习构建一个物理世界的虚拟沙盒，让 AI 在沙盒里低成本试错。
#### Dreamer 系列 (V1-V3) (Hafner et al.):
#### OpenVLA (2024)


## 阶段五 Ai Agent
- https://www.runoob.com/ai-agent/ai-agent-tutorial.html




#### 视觉编码器与大语言模型结合方法
- 视觉编码器提取图像特征 → 投影层(Projector)对齐维度 → 大语言模型生成文本
- 典型结构：`ViT + MLP Projector + LLM (如Vicuna/Llama)`

#### 多模态提示工程
- 设计包含图像描述、区域框、指令的提示模板
- 示例：`"<image>\nUser: 描述这张图片中的人物动作。\nAssistant:"`




## 2026年多模态AI实现了质的飞跃，核心转变是从"看图说话"进化到"看懂世界"。北京智源研究院明确指出，AI正从NTP（Next Token Prediction，预测下一个词）范式转向NSP（Next-State Prediction，预测世界状态）范式，模型开始理解物理世界的规律、因果关系和时空连续性。
## 物理AI（Physical AI）是当前AI领域最受关注的方向之一，被视为继感知AI、生成式AI、Agentic AI之后的第四波AI浪潮。
