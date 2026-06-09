1.环境配置：
环境：
PyTorch  2.3.0
CUDA 12.1
PyTorch / 2.3.0 / 3.12(ubuntu22.04) / 12.1
conda create --name Qwen_env
pip install torchvision==0.20.0 -f https://download.pytorch.org/whl/torch_stable.html
或者
pip install torch==2.3.0+cu121 torchvision==0.20.0+cu124 -f https://download.pytorch.org/whl/torch_stable.html
conda create --name Qwen_env
conda activate Qwen_env
模型下载脚本（这一步需要记录下载的位置）：
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2-VL-7B-Instruct')



2.模型部署
pip install modelscope
#模型下载（这一步需要记录下载的位置）

项目代码地址：github：https://github.com/QwenLM/Qwen3-VL

在Qwen文件夹下：
pip install transformers
pip install 'accelerate>=0.26.0'
pip install qwen-vl-utils[decord]
git clone https://github.com/QwenLM/Qwen2-VL


modelscope项目主页：https://www.modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct
在主页可以找到模型下载方式：
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen3-VL-8B-Instruct')
（如果用autodl可以使用source /etctnetwork_turbo加速）

这里把下载好的模型放在Qwen文件夹下（例如mv /root/.cache/modelscope/hub/Qwen/Qwen2-VL-7B-Instruct /root/Qwen）
可通过du-h看文件夹有没有移动好，没有移动好需要重新移动好


3.可以执行脚本/test.py识别一张图片试试;Qwen3-VL-8B-Instruct自带测试照片
参考：https://blog.csdn.net/WhiffeYF/article/details/144880180中的脚本，修改文件地址即可；

4.LLaMa Factory微调：
安装环境：
source /etc/network_turbo
git clone https://github.com/hiyouga/LLaMA-Factory.gi
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
 检查环境是否安装成功：
llamafactory-cli version

4.1数据集准备
下载数据集
转换dataset.json格式类似于：
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
  },

4.2训练
脚本：llamafactory-cli train examples/train_lora/qwen3moe_lora_sft_kt.yaml
可视化：GRADIO_SERVER_PORT=6006  llamafactory-cli webui并打开


