FROM continuumio/miniconda3:latest

WORKDIR /app

# 安装依赖 - 合并多个RUN命令减少层数
RUN apt-get update && \
    apt-get install -y --no-install-recommends git-lfs ffmpeg && \
    git lfs install && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p pretrained_models

# 创建conda环境并安装依赖 - 合并相关命令
COPY requirements.txt .
RUN conda create -n sparktts python=3.12 -y && \
    /opt/conda/bin/activate sparktts && \
    pip install -r requirements.txt && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('SparkAudio/Spark-TTS-0.5B', local_dir='pretrained_models/Spark-TTS-0.5B')"

# 配置shell和环境激活
SHELL ["/bin/bash", "--login", "-c"]

# 拷贝应用代码 - 放在依赖安装后以利用缓存
COPY . .

# 暴露端口
EXPOSE 2333

# 启动命令
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["source /opt/conda/etc/profile.d/conda.sh && conda activate sparktts && python app.py"]