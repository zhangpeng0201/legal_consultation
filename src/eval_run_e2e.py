'''
端到端评测脚本（检索 → 判定 → 生成）

用法（在 src 目录下）：
    python eval_run_e2e.py --limit 3        # 冒烟：只跑前 3 条，先验证 API/解析正常
    python eval_run_e2e.py                  # 全量 60 条
    python eval_run_e2e.py --ids z01,s11    # 只跑指定题号
    python eval_run_e2e.py --no-answer      # 只测判定，不调生成（省一半调用）

做什么：
    1. 读评测集（z=条号直查 / s=口语描述 / n=应拒答）
    2. 每条走真实链路：get_search_result -> query_judge ->（正例且判定为真时）generate_answer
    3. 三层指标：
       - 判定层：正例判真率 / 拒答正确率 / 误拒率 / 误答率
       - 引用层：judge 给出的 evidence 是否命中 must_hit（宽容匹配法名）
       - 生成层：答案是否带条号引用、引用是否落在 must_hit 内、篇幅是否达标
    4. 落盘逐条明细与配置指纹

注意：
    - 走的是生产参数（rrf_search 默认 top_k=5），所以判定层天花板 = 检索层 Hit@5
    - 会真实调用大模型：正例 2 次/条、负例 1 次/条
    - 评测集里 borderline=true 的题「可否由条文回答」有解释空间，主指标排除、单独列出
    - 篇幅按 prompt 口径计：剔除【法律依据】整段（条文原文不计入）后再数字数
'''

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

from config import BASE_DIR, OUT_DIR, FILE_NAME
from acc_index import cn_to_int
from llm import get_search_result, query_judge, generate_answer

DEFAULT_SET = BASE_DIR / 'data' / 'eval' / 'golden_set.json'
TOP_K = 5                      # 与 llm.get_search_result 内部一致

EV_RE = re.compile(r'第([零一二三四五六七八九十百千]+)条')
SUP_RE = re.compile(r'^之([零一二三四五六七八九十百千]+)')
SEC_RE = re.compile(r'\*\*(问题结论|法律依据|简要分析)\*\*[：:]')
BODY_LIMIT = 200               # prompt 写的是「总字数（不含条文原文）不超过 200 字」


def answer_body_len(answer):
    '''
    数「不含条文原文」的字数：按三段小标题切分，跳过【法律依据】整段。
    结构不符预期时退回整段长度（宁可高估，不虚报达标）。
    '''
    parts = SEC_RE.split(answer or '')
    total = 0
    for i in range(1, len(parts) - 1, 2):
        if parts[i] != '法律依据':
            total += len(parts[i + 1].strip())
    return total or len(answer or '')


def sha12(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:12]


def stat_of(path):
    p = Path(path)
    if not p.exists():
        return None
    s = p.stat()
    return [s.st_size, int(s.st_mtime)]


def build_fingerprint():
    '''配置指纹：模型 + 生成参数 + 语料 + 索引 + 源码（不含评测脚本自身）'''
    src_dir = Path(__file__).resolve().parent
    exclude = {Path(__file__).name, 'eval_run.py'}
    return {
        'model': os.environ.get('DEEPSEEK_NAME'),
        'params': {'temperature': 0, 'max_tokens': 800, 'thinking': 'disabled'},
        'top_k': TOP_K,
        'corpus': stat_of(FILE_NAME),
        'index_bm25': stat_of(OUT_DIR / 'bm25_index.pkl'),
        'index_vecs': stat_of(OUT_DIR / 'embeddings' / 'vecs.npy'),
        'src_hash': {p.name: sha12(p) for p in sorted(src_dir.glob('*.py'))
                     if p.name not in exclude},
    }


def load_set(path, ids=None):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    items = data['items']
    if ids:
        want = {x.strip() for x in ids.split(',') if x.strip()}
        items = [i for i in items if i['id'] in want]
        assert items, f'--ids 没匹配到任何题目: {ids}'
    return data, items


def parse_evidence(evidence):
    '''把 "《中华人民共和国民法典》第一千二百五十四条" 拆成 {title, article, sup}'''
    parsed = []
    for s in evidence or []:
        if not isinstance(s, str):
            continue
        t = s.replace('《', '').replace('》', '').replace(' ', '').replace('\u3000', '')
        m = EV_RE.search(t)
        if not m:
            parsed.append({'raw': s, 'title': None, 'article': None, 'sup': 0})
            continue
        sup = SUP_RE.match(t[m.end():])
        parsed.append({
            'raw': s,
            'title': t[:m.start()],
            'article': cn_to_int(m.group(1)),
            'sup': cn_to_int(sup.group(1)) if sup else 0,
        })
    return parsed


def title_match(got, must):
    '''宽容匹配法名：全称一致，或一方是另一方的后缀（"劳动合同法" vs "中华人民共和国劳动合同法"）'''
    if not got:
        return False
    if got == must:
        return True
    return len(got) >= 3 and (must.endswith(got) or got.endswith(must))


def cite_of(parsed, must_hit):
    '''返回 (命中哪些 must_hit, 覆盖全部了没)'''
    hit = []
    for title, article in must_hit:
        for p in parsed:
            if p['article'] == article and title_match(p['title'], title):
                hit.append([title, article])
                break
    return hit, len(hit) == len(must_hit)


def call_with_retry(fn, *args, tries=3, **kwargs):
    '''返回 (结果, 错误信息)；解析失败/网络抖动都重试'''
    last = None
    for i in range(tries):
        try:
            return fn(*args, **kwargs), None
        except Exception as e:
            last = e
            if i < tries - 1:
                time.sleep(2 + 3 * i)
    return None, f'{type(last).__name__}: {last}'


def as_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() == 'true'
    return bool(x)


def run_one(item, with_answer=True):
    '''跑一条，返回明细记录'''
    rec = {
        'id': item['id'],
        'type': item['type'],
        'query': item['query'],
        'expected_answerable': item['answerable'],
        'must_hit': item['must_hit'],
        'borderline': bool(item.get('borderline')),   # 可否由条文回答存在争议，主指标排除
    }
    t0 = time.time()

    context = get_search_result(item['query'])
    judge, err = call_with_retry(query_judge, item['query'], context)
    rec['judge_sec'] = round(time.time() - t0, 2)

    if err:
        rec.update({'ok': False, 'error': err})
        return rec

    answerable = as_bool(judge.get('answerable'))
    evidence = judge.get('evidence') or []
    parsed = parse_evidence(evidence)
    rec.update({
        'ok': True,
        'judge_answerable': answerable,
        'judge_reason': judge.get('reason'),
        'evidence_raw': evidence,
        'evidence_parsed': parsed,
        'judge_correct': answerable == item['answerable'],
    })

    if item['answerable']:
        hit, full = cite_of(parsed, item['must_hit'])
        rec['cite_hit'] = hit
        rec['cite_hit_any'] = bool(hit)
        rec['cite_hit_full'] = full

    if answerable and with_answer:
        t1 = time.time()
        answer, err2 = call_with_retry(generate_answer, item['query'], context)
        rec['gen_sec'] = round(time.time() - t1, 2)
        if err2:
            rec['gen_ok'] = False
            rec['gen_error'] = err2
        else:
            rec['gen_ok'] = True
            rec['answer_len'] = len(answer)                    # 整段（含条文原文）
            rec['body_len'] = answer_body_len(answer)          # 不含条文原文，与 prompt 口径一致
            rec['body_in_limit'] = rec['body_len'] <= BODY_LIMIT
            rec['answer'] = answer[:1200]
            # 生成层引用检查：答案里出现的条号
            cited = [(cn_to_int(m.group(1))) for m in EV_RE.finditer(answer)]
            rec['answer_articles'] = sorted(set(cited))
            if item['must_hit']:
                want = {a for _, a in item['must_hit']}
                rec['answer_cite_hit'] = bool(want & set(cited))

    rec['total_sec'] = round(time.time() - t0, 2)
    return rec


def pct(a, b):
    return round(a / b, 4) if b else None


def summarize(records):
    ok = [r for r in records if r.get('ok')]
    main = [r for r in ok if not r.get('borderline')]      # 主指标集：排除争议样例
    border = [r for r in ok if r.get('borderline')]
    pos = [r for r in main if r['expected_answerable']]
    neg = [r for r in main if not r['expected_answerable']]

    pos_true = sum(1 for r in pos if r['judge_answerable'])
    neg_reject = sum(1 for r in neg if not r['judge_answerable'])
    cite_any = sum(1 for r in pos if r.get('cite_hit_any'))
    cite_full = sum(1 for r in pos if r.get('cite_hit_full'))
    gen_rows = [r for r in pos if r.get('gen_ok')]
    gen_cite = sum(1 for r in gen_rows if r.get('answer_cite_hit'))
    gen_in_limit = sum(1 for r in gen_rows if r.get('body_in_limit'))

    by_type = {}
    for r in main:
        b = by_type.setdefault(r['type'], {'n': 0, 'judge_correct': 0,
                                           'judge_true': 0, 'cite_any': 0, 'pos': 0})
        b['n'] += 1
        b['judge_correct'] += 1 if r['judge_correct'] else 0
        if r['expected_answerable']:
            b['pos'] += 1
            b['judge_true'] += 1 if r['judge_answerable'] else 0
            b['cite_any'] += 1 if r.get('cite_hit_any') else 0

    for t, b in by_type.items():
        b['judge_correct_rate'] = pct(b['judge_correct'], b['n'])
        if b['pos']:
            b['judge_true_rate'] = pct(b['judge_true'], b['pos'])
            b['cite_any_rate'] = pct(b['cite_any'], b['pos'])

    return {
        'judge': {
            'pos_n': len(pos),
            'pos_judged_true': pos_true,
            'pos_judged_true_rate': pct(pos_true, len(pos)),
            'false_reject': len(pos) - pos_true,
            'false_reject_rate': pct(len(pos) - pos_true, len(pos)),
            'neg_n': len(neg),
            'neg_rejected': neg_reject,
            'neg_reject_rate': pct(neg_reject, len(neg)),
            'false_accept': len(neg) - neg_reject,
            'false_accept_rate': pct(len(neg) - neg_reject, len(neg)),
            'accuracy': pct(pos_true + neg_reject, len(pos) + len(neg)),
        },
        'cite': {
            'pos_n': len(pos),
            'cite_any': cite_any,                         # 主口径：命中 must_hit 任一条即算对
            'cite_any_rate': pct(cite_any, len(pos)),
            'multi_item_n': sum(1 for r in pos if len(r['must_hit']) > 1),   # 这几条 must_hit 是"二选一"
            'cite_full': cite_full,                       # 诊断用：多选一的题天然到不了 100%
            'cite_full_rate': pct(cite_full, len(pos)),
        },
        'borderline': [{
            'id': r['id'],
            'query': r['query'],
            'expected': r['expected_answerable'],
            'judge_answerable': r['judge_answerable'],
            'agree': r['judge_correct'],
            'reason': r.get('judge_reason'),
        } for r in border],
        'generate': {
            'n': len(gen_rows),
            'cite_hit': gen_cite,
            'cite_hit_rate': pct(gen_cite, len(gen_rows)),
            'avg_len': round(sum(r['answer_len'] for r in gen_rows) / len(gen_rows)) if gen_rows else None,
            'avg_body_len': round(sum(r['body_len'] for r in gen_rows) / len(gen_rows)) if gen_rows else None,
            'body_limit': BODY_LIMIT,
            'in_limit': gen_in_limit,                 # 主口径：不含条文原文 <= 200 字
            'in_limit_rate': pct(gen_in_limit, len(gen_rows)),
        },
        'by_type': by_type,
        'errors': [{'id': r['id'], 'error': r.get('error') or r.get('gen_error')}
                   for r in records if not r.get('ok') or r.get('gen_ok') is False],
    }


def run(set_path, out_path, limit=None, ids=None, with_answer=True):
    data, items = load_set(set_path, ids)
    if limit:
        items = items[:limit]

    records = []
    t0 = time.time()
    total = len(items)
    for n, item in enumerate(items, 1):
        rec = run_one(item, with_answer)
        records.append(rec)
        flag = ''
        if rec.get('ok'):
            flag = '✔' if rec['judge_correct'] else '✘'
        else:
            flag = 'ERR'
        print(f"[{n}/{total}] {rec['id']:>4} {rec['type']} {flag} "
              f"{rec['query'][:28]} ({rec.get('total_sec', 0)}s)", flush=True)

    metrics = summarize(records)
    report = {
        'meta': {
            'set': str(set_path),
            'set_version': data.get('meta', {}).get('version'),
            'top_k': TOP_K,
            'with_answer': with_answer,
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_sec': round(time.time() - t0, 2),
            'counts': {'total': total,
                       'positive': sum(1 for i in items if i['answerable']),
                       'negative': sum(1 for i in items if not i['answerable'])},
            'fingerprint': build_fingerprint(),
        },
        'metrics': metrics,
        'details': records,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    j, c, g = metrics['judge'], metrics['cite'], metrics['generate']
    print('\n===== 端到端结果 =====')
    print(f"判定层  正例判真 {j['pos_judged_true']}/{j['pos_n']} ({j['pos_judged_true_rate']:.1%})  "
          f"误拒 {j['false_reject']} 条 ({j['false_reject_rate']:.1%})")
    print(f"        负例拒答 {j['neg_rejected']}/{j['neg_n']} ({j['neg_reject_rate']:.1%})  "
          f"误答 {j['false_accept']} 条")
    print(f"        整体判定准确率 {j['accuracy']:.1%}")
    print(f"引用层  evidence 命中 must_hit {c['cite_any']}/{c['pos_n']} ({c['cite_any_rate']:.1%})"
          f"（其中 {c['multi_item_n']} 条是二选一，覆盖全部 {c['cite_full']} 条，仅作诊断）")
    print(f"生成层  答案引用命中 {g['cite_hit']}/{g['n']} ({g['cite_hit_rate']:.1%})  "
          f"不含原文均 {g['avg_body_len']} 字 / 整段均 {g['avg_len']} 字")
    print(f"        篇幅达标（不含条文原文 ≤{g['body_limit']} 字）{g['in_limit']}/{g['n']} "
          f"({g['in_limit_rate']:.1%})")
    for t, b in metrics['by_type'].items():
        print(f"  {t}: n={b['n']} 判定正确 {b['judge_correct']}/{b['n']} ({b['judge_correct_rate']:.0%})")
    if metrics['borderline']:
        bl = metrics['borderline']
        print(f"边界样例 {len(bl)} 条（不计入主指标）: " +
              ' | '.join(f"{x['id']} 判{'真' if x['judge_answerable'] else '假'}"
                         f"({'一致' if x['agree'] else '不一致'})" for x in bl))
    if metrics['errors']:
        print(f"调用失败 {len(metrics['errors'])} 条: {metrics['errors']}")
    print(f"耗时 {report['meta']['elapsed_sec']}s · 明细已写入 {out_path}")
    return report


def main():
    ap = argparse.ArgumentParser(description='法询端到端评测（判定 + 引用 + 生成）')
    ap.add_argument('--set', type=Path, default=DEFAULT_SET, help='评测集路径')
    ap.add_argument('--out', type=Path, default=None, help='明细输出路径')
    ap.add_argument('--limit', type=int, default=None, help='只跑前 N 条（冒烟用）')
    ap.add_argument('--ids', type=str, default=None, help='只跑指定题号，逗号分隔')
    ap.add_argument('--no-answer', action='store_true', help='不调生成，只测判定')
    args = ap.parse_args()

    out = args.out or OUT_DIR / f"eval_e2e_{time.strftime('%Y%m%d_%H%M%S')}.json"
    run(args.set, out, args.limit, args.ids, with_answer=not args.no_answer)


if __name__ == '__main__':
    main()
