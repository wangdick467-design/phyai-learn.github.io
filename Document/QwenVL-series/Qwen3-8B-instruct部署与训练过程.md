
# Qwen-VL 模型部署与 LLaMA-Factory 微调完整流程

## 1. 环境配置

### 创建 Conda 环境
```bash
conda create -n Qwen_env python=3.12
conda activate Qwen_env
```

### 安装 PyTorch（CUDA 12.1）
```bash
# 方式一
pip install torchvision==0.20.0 -f https://download.pytorch.org/whl/torch_stable.html

# 方式二（推荐）
pip install torch==2.3.0+cu121 torchvision==0.20.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

## 2. 模型下载

### 使用 ModelScope 下载模型
```python
from modelscope import snapshot_download

# 下载 Qwen2-VL-7B-Instruct（示例）
model_dir = snapshot_download('Qwen/Qwen2-VL-7B-Instruct')

# 或下载 Qwen3-VL-8B-Instruct
model_dir = snapshot_download('Qwen/Qwen3-VL-8B-Instruct')
```

**加速下载（AutoDL环境）**：
```bash
source /etc/network_turbo
```

**移动模型到工作目录**（推荐放在项目文件夹下）：
```bash
# 示例路径
mv /root/.cache/modelscope/hub/Qwen/Qwen2-VL-7B-Instruct /root/Qwen/
# 或 Qwen3-VL-8B-Instruct

# 检查移动是否成功
du -sh /root/Qwen/Qwen*-VL-*
```

## 3. 模型部署

### 安装依赖
```bash
pip install modelscope
pip install transformers
pip install 'accelerate>=0.26.0'
pip install qwen-vl-utils[decord]
```

### 克隆官方代码
```bash
git clone https://github.com/QwenLM/Qwen2-VL.git
cd Qwen2-VL
```

> **提示**：模型文件建议放在 `Qwen/` 文件夹下，便于后续调用。

## 4. 模型测试

在项目目录下创建 `test.py`，参考以下脚本（可修改图片路径）：

```python
# test.py 示例
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# 加载模型
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/root/Qwen/Qwen2-VL-7B-Instruct",  # 修改为实际模型路径
    torch_dtype="auto",
    device_map="auto"
)

processor = AutoProcessor.from_pretrained("/root/Qwen/Qwen2-VL-7B-Instruct")

# 准备消息（支持多图）
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "path/to/your/image.jpg"},  # 修改图片路径
            {"type": "text", "text": "请描述这张图片的内容。"},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
)
inputs = inputs.to("cuda")

# 生成
generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0])
```

**运行测试**：
```bash
python test.py
```

> Qwen3-VL-8B-Instruct 自带测试图片，可直接使用官方示例。

## 5. LLaMA-Factory 微调

### 5.1 安装 LLaMA-Factory
```bash
source /etc/network_turbo
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

### 检查安装
```bash
llamafactory-cli version
```

### 5.2 数据集准备

将数据集转换为 **ShareGPT** 格式，`dataset_info.json` 示例配置：

```json
{
  "Bridge_Behavior": {
    "file_name": "Bridge_Behavior.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  }
}
```

### 5.3 训练

使用配置文件训练（示例：`qwen3moe_lora_sft_kt.yaml`）：

```bash
llamafactory-cli train /path/to/train_lora/qwen3moe_lora_sft_kt.yaml
```

### 5.4 可视化训练（Web UI）
```bash
GRADIO_SERVER_PORT=6006 llamafactory-cli webui
```

然后在浏览器中打开对应端口查看训练进度。

---

## 常见注意事项

- **模型路径**：全程统一使用绝对路径，避免找不到模型。
- **显存占用**：7B/8B 模型 LoRA 微调建议使用 A100 或更高显存卡。
- **数据集格式**：多模态数据需确保 `images` 字段正确指向图片路径。
- **加速**：AutoDL 等平台记得使用 `network_turbo` 加速下载和安装。
- **参考资料**：
  - Qwen2-VL 官方仓库：https://github.com/QwenLM/Qwen2-VL
  - ModelScope 主页：https://www.modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct
  - 测试脚本参考：https://blog.csdn.net/WhiffeYF/article/details/144880180

**完成以上步骤即可完成 Qwen-VL 模型的部署与自定义微调。**
升级到 SuperGrok
Qwen-VL Deployment and LLaMA-Factory Fine-Tuning Guide - Grok
