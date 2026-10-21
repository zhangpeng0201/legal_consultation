'''
离线评测脚本（检索层）

用法（在 src 目录下）：
    python eval_run.py                    # 默认用 data/eval/golden_set.json，top_k=10
    python eval_run.py --topk 5           # 改召回深度
    python eval_run.py --set ../data/eval/golden_set.json --out ../output/eval_x.json

做什么：
    1. 读评测集（z=条号直查 / s=口语描述 / n=应拒答）
    2. 对每条正例题跑 rrf_search，算首个正确条文的排名
    3. 输出 Hit@1 / Hit@5 / Hit@10 / MRR，并按类型分别统计
    4. 落盘逐条明细（含未召回的题和它实际召回了什么），并记录配置指纹

注意：
    - 只评测检索层，不调用大模型，零 API 成本
    - 负例（n）不参与检索指标；它们留给端到端评测判拒答
'''

import argparse
import hashlib
import json
import time
from pathlib import Path

from config import BASE_DIR, OUT_DIR, FILE_NAME
from rrf_search import rrf_search, CHANNELS_WEIGHT, K

DEFAULT_SET = BASE_DIR / 'data' / 'eval' / 'golden_set.json'
PER_RECALL = 10          # 每条查询固定看前 10 条，Hit@k 由排名换算，不用重复检索


def sha12(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:12]


def stat_of(path):
    '''返回 (字节数, 修改时间)，用来做资产指纹'''
    p = Path(path)
    if not p.exists():
        return None
    s = p.stat()
    return [s.st_size, int(s.st_mtime)]


def build_fingerprint():
    '''配置指纹：参数 + 资产 + 源码，任一变化都会体现出来'''
    src_dir = Path(__file__).resolve().parent
    return {
        'embed_model': 'bge-small-zh-v1.5 (512d, cpu)',
        'rrf_k': K,
        'channel_weights': [list(x) for x in CHANNELS_WEIGHT],
        'corpus': stat_of(FILE_NAME),
        'index_bm25': stat_of(OUT_DIR / 'bm25_index.pkl'),
        'index_vecs': stat_of(OUT_DIR / 'embeddings' / 'vecs.npy'),
        'src_hash': {p.name: sha12(p) for p in sorted(src_dir.glob('*.py'))
                     if p.name != Path(__file__).name},   # 排除本脚本，避免改评测代码本身触发指纹变化
    }


def load_set(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    items = data['items']
    ids = [i['id'] for i in items]
    assert len(ids) == len(set(ids)), '评测集里 id 重复'
    for i in items:
        assert i['type'] in ('条号直查', '口语描述', '应拒答'), f"{i['id']} 类型非法: {i['type']}"
        if i['answerable']:
            assert i['must_hit'], f"{i['id']} 是正例但没有 must_hit"
        else:
            assert not i['must_hit'], f"{i['id']} 是负例却带了 must_hit"
    return data


def match_rank(result, must_hit):
    '''在检索结果里找第一个命中 must_hit 的位置，返回 1-based 排名或 None'''
    for rank, item in enumerate(result, 1):
        key = item.get('key') or ()
        if len(key) < 2:
            continue
        for title, article in must_hit:
            if key[0] == title and key[1] == article:
                return rank
    return None


def run(set_path, top_k, out_path, verbose=True):
    data = load_set(set_path)
    items = data['items']
    positives = [i for i in items if i['answerable']]
    negatives = [i for i in items if not i['answerable']]

    details, missed = [], []
    by_type = {}
    t0 = time.time()

    for i in positives:
        res = rrf_search(i['query'], top_k=PER_RECALL)
        rank = match_rank(res, i['must_hit'])
        rec = {
            'id': i['id'],
            'type': i['type'],
            'query': i['query'],
            'rank': rank,
            'hit_at_k': (rank is not None and rank <= top_k),
            'top3': [{'key': list(x['key'][:2]), 'score': round(x['score'], 6)} for x in res[:3]],
        }
        details.append(rec)

        b = by_type.setdefault(i['type'], {'n': 0, 'ranks': []})
        b['n'] += 1
        b['ranks'].append(rank)
        if rank is None:
            missed.append(i['id'])

    def summarize(rows):
        n = len(rows)
        hit = sum(1 for r in rows if r['rank'] is not None)
        h1 = sum(1 for r in rows if r['rank'] == 1)
        hk = sum(1 for r in rows if r['rank'] is not None and r['rank'] <= top_k)
        rr = sum(1.0 / r['rank'] for r in rows if r['rank'] is not None)
        return {
            'n': n,
            'hit': hit,
            'hit_rate': round(hit / n, 4) if n else None,
            'hit@1': h1,
            f'hit@{top_k}': hk,
            f'hit@{top_k}_rate': round(hk / n, 4) if n else None,
            'mrr': round(rr / n, 4) if n else None,
        }

    overall = summarize(details)
    type_summary = {
        t: {
            'n': b['n'],
            'hit': sum(1 for r in b['ranks'] if r is not None),
            'hit_rate': round(sum(1 for r in b['ranks'] if r is not None) / b['n'], 4),
            'hit@1': sum(1 for r in b['ranks'] if r == 1),
            f'hit@{top_k}': sum(1 for r in b['ranks'] if r is not None and r <= top_k),
            f'hit@{top_k}_rate': round(sum(1 for r in b['ranks'] if r is not None and r <= top_k) / b['n'], 4),
            'mrr': round(sum(1.0 / r for r in b['ranks'] if r is not None) / b['n'], 4),
        }
        for t, b in by_type.items()
    }

    report = {
        'meta': {
            'set': str(set_path),
            'set_version': data.get('meta', {}).get('version'),
            'top_k': top_k,
            'recall_depth': PER_RECALL,
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_sec': round(time.time() - t0, 2),
            'counts': {'total': len(items), 'positive': len(positives), 'negative': len(negatives)},
            'fingerprint': build_fingerprint(),
        },
        'metrics': {'overall': overall, 'by_type': type_summary},
        'missed': missed,
        'details': details,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"评测集 {set_path.name} · 正例 {len(positives)} 条 · 负例 {len(negatives)} 条 · top_k={top_k}")
        print(f"总体   Hit@1 {overall['hit@1']}/{overall['n']}  "
              f"Hit@{top_k} {overall[f'hit@{top_k}']}/{overall['n']} "
              f"({overall[f'hit@{top_k}_rate']:.1%})  MRR {overall['mrr']:.4f}")
        for t, s in type_summary.items():
            print(f"  {t}  Hit@1 {s['hit@1']}/{s['n']}  召回 {s['hit']}/{s['n']} ({s['hit_rate']:.0%})  MRR {s['mrr']:.4f}")
        print(f"未召回 {len(missed)} 条: {missed if missed else '无'}")
        print(f"耗时 {report['meta']['elapsed_sec']}s · 明细已写入 {out_path}")
    return report


def main():
    ap = argparse.ArgumentParser(description='法询检索层离线评测')
    ap.add_argument('--set', type=Path, default=DEFAULT_SET, help='评测集路径')
    ap.add_argument('--topk', type=int, default=10, help='Hit@k 的 k（召回深度固定 10）')
    ap.add_argument('--out', type=Path, default=None, help='明细输出路径')
    args = ap.parse_args()

    out = args.out or OUT_DIR / f"eval_retrieval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    run(args.set, args.topk, out)


if __name__ == '__main__':
    main()
