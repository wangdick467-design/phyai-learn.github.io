# phyai-learn.github.io
Share ai learning💡
---
## 主要是多模态大模型的学习笔记和复现。物理学本，top2电子学博士想试试Ai时代0基础自学Ai。
## 2026年多模态AI实现了质的飞跃，核心转变是从"看图说话"进化到"看懂世界"。北京智源研究院明确指出，AI正从NTP（Next Token Prediction，预测下一个词）范式转向NSP（Next-State Prediction，预测世界状态）范式，模型开始理解物理世界的规律、因果关系和时空连续性。
## 物理AI（Physical AI）是当前AI领域最受关注的方向之一，被视为继感知AI、生成式AI、Agentic AI之后的第四波AI浪潮。
🗓️ 详细学习计划
阶段一：打基础 (1-2个月)
1. 编程基础

Python: 基础语法、面向对象、常用库

数据处理: NumPy, Pandas, Matplotlib

数据处理能力: 能够处理图像、文本、音频数据

2. 数学基础

线性代数: 向量、矩阵、张量运算

概率统计: 概率分布、期望、方差、贝叶斯定理

微积分: 梯度、导数、链式法则

3. 深度学习框架

PyTorch​ (推荐): 张量操作、自动求导、神经网络模块

TensorFlow: 作为备选学习

4. 单模态基础

视觉: CNN原理、ResNet、图像分类/检测 (参考 Stanford CS231n
)

文本: Transformer架构、自注意力机制 (参考 Stanford CS224N
)

音频: 频谱图、MFCC特征、音频分类

阶段二：理解多模态核心原理 (1个月)
1. 跨模态对齐

对比学习原理

CLIP模型的图文对齐机制

跨模态表示学习

2. 融合策略

早期融合 (Early Fusion)

晚期融合 (Late Fusion)

联合融合 (Joint Fusion)

注意力融合

3. 经典模型

CLIP: 图文对比预训练

BLIP: 引导语言图像预训练

ViLT: 视觉语言Transformer

LXMERT: 学习跨模态编码器表示

4. 大型多模态语言模型

LLaVA: 视觉指令调优

视觉编码器与大语言模型结合方法

多模态提示工程

阶段三：动手实战 (持续进行)
1. 工具链掌握

Hugging Face Transformers: 预训练模型使用与微调

OpenMMLab/MMF: 多模态框架

其他相关工具库

2. 常用数据集实践

COCO: 图文对齐、目标检测

VQA v2: 视觉问答

MSR-VTT: 视频字幕生成

Flickr30k, Conceptual Captions

3. 实战项目

图像描述生成: CNN + RNN/Transformer架构

视觉问答: 多模态注意力机制

文本-图像检索: 双编码器架构

多模态情感分析: 视觉+文本情感识别

多模态Agent: 使用LangChain构建图文问答系统

阶段四：跟进前沿 (持续)
1. 关注顶会论文

CVPR, ICCV, ECCV: 计算机视觉顶会

ICML, NeurIPS, ICLR: 机器学习顶会

ACL, EMNLP: 自然语言处理顶会

2. 跟踪开源项目

OpenVLA: 具身智能

LLaVA系列: 开源多模态对话

MiniCPM: 轻量多模态模型

其他活跃开源社区

3. 探索新兴方向

世界模型: 环境建模与预测

具身智能: 机器人多模态感知

多模态Agent: 自主任务完成

视频理解、3D视觉语言
└── README.md                     # 本文件
