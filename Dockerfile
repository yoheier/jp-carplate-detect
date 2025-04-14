# Raspberry Pi ARM64 Debian Slim イメージ（Python 3.11）
FROM python:3.11-slim-bullseye

# 環境変数（OpenMP で CPU 並列強化）
ENV OMP_NUM_THREADS=6
ENV PYTHONUNBUFFERED=1

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ
WORKDIR /workspace

# ソースコードとホイールをコピー
COPY . /workspace
COPY ./onnxruntime-1.22.0-cp311-cp311-linux_aarch64.whl /workspace/

# Python パッケージインストール（requirements.txt 使用、numpy は 1.24 固定）
RUN pip install --upgrade pip && \
    pip install numpy==1.24 && \
    pip install ./onnxruntime-1.22.0-cp311-cp311-linux_aarch64.whl && \
    pip install -r requirements.txt

# エントリーポイント
CMD ["python3", "main_pipeline.py"]

