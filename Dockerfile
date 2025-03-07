FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app
# 设置非交互式前端，防止安装过程中的提示
ENV DEBIAN_FRONTEND=noninteractive
# Install git and git-lfs (for model download)
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install git-lfs
RUN git lfs install

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download the model (you can either do this here or mount it as a volume)
RUN mkdir -p pretrained_models && \
    git clone https://huggingface.co/SparkAudio/Spark-TTS-0.5B pretrained_models/Spark-TTS-0.5B

# Copy code
COPY . .

# Set CMD to run the handler
CMD ["python", "handler.py"]