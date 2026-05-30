#!/bin/bash

# CosyVoice 一键安装/配置环境脚本
# 使用方法: bash install.sh

set -e

echo "=========================================="
echo "CosyVoice 环境安装脚本"
echo "=========================================="

# 1. 初始化 git submodule
echo "[1/10] 初始化 git submodule..."
git submodule update --init --recursive

# 2. 创建 conda 环境
echo "[2/10] 创建 conda 环境 (cosyvoice, python=3.10)..."
conda create -n cosyvoice -y python=3.10

# 3. 激活 conda 环境
echo "[3/10] 激活 conda 环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cosyvoice

# 4. 安装 CUDA 工具包
echo "[4/10] 安装 CUDA 工具包 (cudatoolkit, cudnn)..."
conda install -c conda-forge cudatoolkit cudnn -y

# 5. 安装 CUDA NVCC
echo "[5/10] 安装 CUDA NVCC..."
conda install -c nvidia cuda-nvcc -y

# 6. 安装 requirements.txt (先安装主要依赖，不安装 whisper 和 pyworld)
echo "[6/10] 安装 requirements.txt..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host=pypi.tuna.tsinghua.edu.cn

# 7. 安装 sox
echo "[7/10] 安装 sox 和 libsox-dev..."
sudo apt-get install sox libsox-dev -y

# 8. 解压并安装 ttsfrd 相关依赖
echo "[8/10] 解压并安装 ttsfrd 相关依赖..."
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
cd ../..

# 9. 特殊处理：降级 setuptools -> 安装 whisper/pyworld (无隔离) -> 升级 setuptools
echo "[9/10] 特殊处理：降级 setuptools 以安装 openai-whisper 和 pyworld..."
echo "      降级 setuptools 到 81.0.0..."
pip install "setuptools==81.0.0" -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

echo "      安装 openai-whisper 和 pyworld (无构建隔离)..."
pip install openai-whisper==20231117 pyworld==0.3.4 --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host=pypi.tuna.tsinghua.edu.cn

echo "      升级 setuptools 到最新版本..."
pip install --upgrade setuptools -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host=pypi.tuna.tsinghua.edu.cn

# 10. 下载预训练模型 (在 ttsfrd 安装完成后下载所有模型)
echo "[10/10] 下载预训练模型..."
python3 << 'EOF'
try:
    from modelscope import snapshot_download
    print("使用 ModelScope SDK 下载模型...")
    
    # 下载所有预训练模型
    models = [
        ('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', 'pretrained_models/Fun-CosyVoice3-0.5B'),
        ('iic/CosyVoice2-0.5B', 'pretrained_models/CosyVoice2-0.5B'),
        ('iic/CosyVoice-300M', 'pretrained_models/CosyVoice-300M'),
        ('iic/CosyVoice-300M-SFT', 'pretrained_models/CosyVoice-300M-SFT'),
        ('iic/CosyVoice-300M-Instruct', 'pretrained_models/CosyVoice-300M-Instruct'),
        ('iic/CosyVoice-ttsfrd', 'pretrained_models/CosyVoice-ttsfrd'),
    ]
    
    for model_id, local_dir in models:
        print(f"正在下载 {model_id} -> {local_dir}")
        snapshot_download(model_id, local_dir=local_dir)
        print(f"{model_id} 下载完成")
    
    print("所有模型下载完成!")
except ImportError:
    print("ModelScope 未安装，尝试使用 HuggingFace...")
    try:
        from huggingface_hub import snapshot_download
        
        models = [
            ('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', 'pretrained_models/Fun-CosyVoice3-0.5B'),
            ('FunAudioLLM/CosyVoice2-0.5B', 'pretrained_models/CosyVoice2-0.5B'),
            ('FunAudioLLM/CosyVoice-300M', 'pretrained_models/CosyVoice-300M'),
            ('FunAudioLLM/CosyVoice-300M-SFT', 'pretrained_models/CosyVoice-300M-SFT'),
            ('FunAudioLLM/CosyVoice-300M-Instruct', 'pretrained_models/CosyVoice-300M-Instruct'),
            ('FunAudioLLM/CosyVoice-ttsfrd', 'pretrained_models/CosyVoice-ttsfrd'),
        ]
        
        for model_id, local_dir in models:
            print(f"正在下载 {model_id} -> {local_dir}")
            snapshot_download(model_id, local_dir=local_dir)
            print(f"{model_id} 下载完成")
        
        print("所有模型下载完成!")
    except ImportError:
        print("错误：ModelScope 和 HuggingFace 均未安装，请手动下载模型")
        print("请参考 README.md 中的模型下载说明")
EOF

echo "=========================================="
echo "安装完成!"
echo "请运行以下命令激活环境:"
echo "  conda activate cosyvoice"
echo "=========================================="
