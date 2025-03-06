FROM continuumio/miniconda3:latest

WORKDIR /app

# Install git-lfs and ffmpeg
RUN apt-get update && \
    apt-get install -y git-lfs ffmpeg && \
    git lfs install && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Create and activate conda environment
RUN conda create -n sparktts python=3.12 -y && \
    echo "source /opt/conda/etc/profile.d/conda.sh && conda activate sparktts" > ~/.bashrc

# Install pip requirements
SHELL ["/bin/bash", "--login", "-c"]
RUN conda activate sparktts && \
    pip install -r requirements.txt

# Make directory for models
RUN mkdir -p pretrained_models

# Download the model
RUN conda activate sparktts && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('SparkAudio/Spark-TTS-0.5B', local_dir='pretrained_models/Spark-TTS-0.5B')"

# Alternative model download method (commented out)
# RUN git clone https://huggingface.co/SparkAudio/Spark-TTS-0.5B pretrained_models/Spark-TTS-0.5B

# Copy the rest of your application
COPY . .

# No need to create permanent directories for audio output
# as we'll use Python's tempfile module

# Expose the port the app runs on
EXPOSE 2333

# Set the default command to activate conda environment and launch your application
ENTRYPOINT ["/bin/bash", "--login", "-c"]
CMD ["source /opt/conda/etc/profile.d/conda.sh && conda activate sparktts && python app.py"]