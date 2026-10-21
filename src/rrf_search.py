'''
倒数排序模块
'''

from acc_search import exact_search
from bm25_search import bm25_search
from embedding_search import embed_search


CHANNELS_WEIGHT = [('exact', 3.0), ('bm25', 1.0), ('embedding', 1.0)]
K = 60

def rrf_search(query, top_k=5):
    result = []
    channels_search = {
        'exact': exact_search(query),
        'bm25': bm25_search(query),
        'embedding': embed_search(query)
    }
    scores, l2k = {}, {}
    for channel, weight in CHANNELS_WEIGHT:
        for rank, item in enumerate(channels_search[channel], 1):
            lineno = item.get('lineno')
            if lineno is None:
                continue
            scores[lineno] = scores.get(lineno, 0) + weight / (rank + K)
            l2k[lineno] = item.get('key')
            # print(f'{channel}: {lineno}, {l2k[lineno]}, {scores[lineno]}')
    for lineno, score in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]:
        result.append({
            'channel':'rrf',
            'lineno': lineno,
            'key': l2k[lineno],
            'score': score
        })
    return result


