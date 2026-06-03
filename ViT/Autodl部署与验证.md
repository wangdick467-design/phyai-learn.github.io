#参考：https://developer.baidu.com/article/detail.html?id=5596066
#源码https://github.com/google-research/vision_transformer#running-on-cloud

#环境配置


```bash
# 切换到数据盘（所有大文件都存放于此）
cd /root/autodl-tmp

# 创建独立的 conda 环境（Python 3.10）
conda create -p /root/autodl-tmp/conda/envs/vit_env python=3.10 -y

# 激活环境
conda activate /root/autodl-tmp/conda/envs/vit_env

# 克隆 ViT 官方仓库
git clone https://github.com/google-research/vision_transformer.git

# 进入项目目录
cd vision_transformer

# 安装项目所需的 Python 依赖库
pip install -r vit_jax/requirements.txt

# 验证 JAX 是否安装成功（应显示 GPU 信息）
# If using GPU:
pip install -r vit_jax/requirements.txt 
清除无效的 Git 代理配置
在终端直接运行以下两行命令，强制取消 Git 的全局代理设置：

Bash
git config --global --unset http.proxy
git config --global --unset https.proxy

python -c "import jax; print(jax.devices())"
# 创建模型存放目录
mkdir -p /root/autodl-tmp/vit_models

# 下载 ViT-B/16 预训练权重（391 MB，最常用的版本）
# -c 参数支持断点续传
wget -c https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz \
  -P /root/autodl-tmp/vit_models/



  # 创建日志输出目录
mkdir -p /root/autodl-tmp/vit_logs

# 启动微调（使用本地预训练权重）
python -m vit_jax.main \
  --workdir=/root/autodl-tmp/vit_logs/vit-$(date +%s) \
  --config=$(pwd)/vit_jax/configs/vit.py:b16,cifar10 \
  --config.pretrained_path='/root/autodl-tmp/vit_models/ViT-B_16.npz'
  
