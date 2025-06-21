FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip git && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple

COPY app/ .

ENV MODEL_DIR=model
ENV DEVICE=cuda
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "summarization_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "5"]
