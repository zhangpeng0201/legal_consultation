FROM python:3.11-slim

# 不写 .pyc、pip 不留缓存；离线模式防止 sentence-transformers 联网检查模型
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

# ① 先单独装 CPU 版 torch
#    不装默认版：默认版会拉 2GB+ 的 CUDA 依赖，本项目只用 CPU 推理（model_kwargs={'device':'cpu'}）
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# ② 其余依赖（先装依赖，再 COPY 代码，改代码时不必重装依赖层）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ③ 代码与运行资产（按体积从小到大排，便于层复用）
COPY src/ ./src/
COPY html/ ./html/
COPY data/json/law_name_map.json ./data/json/
COPY data/jsonl/law_corpus.jsonl ./data/jsonl/
COPY output/ ./output/
COPY bge-small-zh-v1.5/ ./bge-small-zh-v1.5/

# 魔搭创空间固定端口 7860
EXPOSE 7860

# --app-dir src：把 src 加入模块搜索路径，使 src 内的 from config import ... 可用
# src/config.py 里 BASE_DIR = Path(__file__).parent.parent = /app，与上面的 COPY 结构对应
CMD ["uvicorn", "service:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "7860"]
