from pathlib import Path
import json


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "jsonl"
FILE_NAME = DATA_DIR / 'law_corpus.jsonl'
OUT_DIR = BASE_DIR / "output"

# 加载数据
def load_data(file_name: str):
    with open(file_name, "r", encoding="utf-8") as f:
        raw = [json.loads(line) for line in f]
    return raw