'''
向量检索模块
'''

from pathlib import Path
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from config import BASE_DIR, OUT_DIR, FILE_NAME, load_data
import numpy as np


KEYS = OUT_DIR / 'bm25_index.pkl'
MODEL = BASE_DIR / 'bge-small-zh-v1.5'
EMBEDDINGS_DIR = OUT_DIR / "embeddings"
EMBEDDINGS_FILE_NAME = EMBEDDINGS_DIR / 'vecs.npy'
_EMB = None
_KEYS = None
_VECS = None


# 构建向量
def build_embeddings():
    global _EMB
    if _EMB is not None:
        return _EMB
    _EMB = HuggingFaceEmbeddings(
        model_name=str(MODEL),
        model_kwargs={'device': 'cpu'},
        encode_kwargs={"normalize_embeddings": True},
    )
    # print('Embeddings built')
    return _EMB

# 加载向量
def load_embeddings():
    global _VECS
    if _VECS is None:
        _VECS = np.load(EMBEDDINGS_FILE_NAME)
    return _VECS


# 保存向量
def save_embeddings(vecs):
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_FILE_NAME, vecs)
    # print('Embeddings saved')

# 向量化文档
def embed_documents():
    data = load_data(FILE_NAME)
    docs = [d['title'] + ' ' + d['section'] for d in data]
    embeddings = build_embeddings()
    vecs = np.asarray(embeddings.embed_documents(docs), dtype=np.float32)
    # print('Embeddings generated')
    return vecs


def get_keys():
    global _KEYS
    if _KEYS is  None:
        with open(KEYS, 'rb') as f:
            _KEYS = pickle.load(f)['keys']
    return _KEYS

# 向量搜索
def embed_search(query):
    result = []
    if not EMBEDDINGS_FILE_NAME.exists():
        raise FileNotFoundError(
            f"向量文件不存在: {EMBEDDINGS_FILE_NAME}\n"
            f"请先运行向量编码生成该文件（见 README「构建索引」），不要在查询时现场编码")
        # vecs = embed_documents()
        # save_embeddings(vecs)
    vecs = load_embeddings()
    # print('Embeddings loaded')
    query = build_embeddings().embed_query(query)
    # print('Query embedded')
    sim_scores = np.dot(vecs, query)
    # print('Similarity scores computed')
    top = np.argpartition(-sim_scores, 10)[:10]
    # print('Top 10 candidates found')
    top = top[np.argsort(-sim_scores[top])].tolist()
    # print('Top 10 candidates sorted')
    keys = get_keys()
    for lineno in top:
        result.append({
            'channel': 'embedding',
            'lineno': lineno,
            'key': keys[lineno],
            'score': float(sim_scores[lineno])
        })
    return result



