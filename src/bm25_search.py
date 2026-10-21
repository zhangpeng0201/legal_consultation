'''
bm25索引模块
'''


import math
from pathlib import Path
import pickle
import jieba
from collections import Counter, defaultdict
from config import FILE_NAME, OUT_DIR, load_data
from acc_index import  get_sec, get_sup


'''
inv   = defaultdict(list)   # 词 → [(行号, 这个词在这条里出现了几次)]
idf   = {}                  # 词 → 权重，越罕见越大
dl    = []                  # dl[i] = 第 i 条有几个词
keys  = []                  # keys[i] = (法名, 条号, 之一)，解析不出存 None
avgdl = 0.0                 # 22260 条的平均词数
N     = 0                   # 一共 22260 条

'''



CACHE = OUT_DIR / 'bm25_index.pkl'
_INDEX = None

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)


# 保存缓存
def save_cache(data, cache_file):
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)
# 加载缓存
def load_cache(cache_file):
    with open(cache_file, "rb") as f:
        return pickle.load(f)

# 构建 bm25 索引
def build_bm25_index(raw):
    docs = [d['title'] + ' ' + d['section'] for d in raw]  # 把文档标题和段落合并
    tokenized_docs = [jieba.lcut_for_search(doc) for doc in docs]  # 对每个文档进行分词
    N = len(tokenized_docs)  # 文档总数
    inv = defaultdict(list)   # 词 → [(行号, 这个词在这条里出现了几次)]，倒排索引
    dl = []                   # dl[i] = 第 i 条有几个词
    keys = []                 # keys[i] = (法名, 条号, 之一)，解析不出存 None
    for lineno, tokens in enumerate(tokenized_docs):
        tf = Counter(tokens)  # 单个文档的词 → 词频
        dl.append(sum(tf.values()))
        for token, count in tf.items():
            inv[token].append((lineno, count))
        # 解析 (法名, 条号, 之一)，与 step06 的 key 格式保持一致
        sec = get_sec(raw[lineno]['section'])
        if sec == -1:
            keys.append((raw[lineno]['title'], None, get_sup(raw[lineno]['section'])))
        else:
            keys.append((raw[lineno]['title'], sec, get_sup(raw[lineno]['section'])))
    avgdl = sum(dl) / N if N else 0.0  # 平均文档分词数
    idf = {token: math.log((N - len(postings) + 0.5) / (len(postings) + 0.5) + 1) for token, postings in inv.items()}  # 逆文档频率
    result = {'inv': dict(inv), 'dl': dl, 'keys': keys, 'avgdl': avgdl, 'N': N, 'idf': idf}
    return result

# 查询 bm25
def query_bm25(index, query, k1=1.5, b=0.75, top_k = 10):
    query_tokens = jieba.lcut_for_search(query)
    scores = defaultdict(float)
    for token in query_tokens:
        if token not in index['inv']:
            continue
        for lineno, tf in index['inv'][token]:
            scores[lineno] += index['idf'][token] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * index['dl'][lineno] / index['avgdl']))
    top = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    result = []
    for lineno, score in top:
        item = {
            "channel": "bm25",
            "lineno": lineno,
            "key": index['keys'][lineno],
            "score": score
        }
        result.append(item)
    return result

def load_or_build(corpus_path=FILE_NAME, path=CACHE):
    corpus, cache = Path(corpus_path), Path(path)
    if cache.exists() and cache.stat().st_mtime > corpus.stat().st_mtime:
        return load_cache(cache)
    index = build_bm25_index(load_data(corpus_path))
    save_cache(index, path)
    return index


# bm25 搜索
def bm25_search(query):
    global _INDEX
    if _INDEX is  None:
        _INDEX = load_or_build()
    return query_bm25(_INDEX, query)




