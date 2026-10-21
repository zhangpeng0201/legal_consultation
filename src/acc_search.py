'''
精确搜索模块：
    输入：《中华人民共和国刑法》第232条
    输出：[
            {
                'channel': 'exact',
                'lineno': 2863,
                'key': ('中华人民共和国刑法', 232, 0),
                'score': 1.0
            }
    ]
'''


import json
import re
from config import BASE_DIR
from acc_index import cn_to_int, law_to_idx


MAPPING_DIR = BASE_DIR / "data" / "json" / "law_name_map.json"
QUERY_RE = re.compile(r"第\s*([零一二三四五六七八九十百千〇\d]+)\s*条"
                      r"(?:之\s*([零一二三四五六七八九十\d]+))?")
LAW_TO_IDX = law_to_idx()

# 加载映射表
def load_mapping():
    with open(MAPPING_DIR, 'r', encoding='utf-8') as f:
        return json.load(f)

MAPPING = load_mapping()

# 解析文本
def parse_text(text):
    title = None
    for key in sorted(MAPPING['table'], key=len, reverse=True):
        if key in text:
            title = MAPPING['table'][key]
            break
    if not title:
        return None
    re_result = QUERY_RE.search(text)
    if not re_result:
        key = (title, None, 0)
        return key
    sec = cn_to_int(re_result.group(1))
    sup = cn_to_int(re_result.group(2)) if re_result.group(2) else 0
    key = (title, sec, sup)
    return key


# 精确搜索
def query(key):
    lineno = LAW_TO_IDX.get(key)
    if lineno is None:
        return [
            {
                "channel": "exact",
                "lineno": None,
                "key": key,
                "score": 0.00
            }
        ]
    return [
        {
            "channel": "exact",
            "lineno": lineno,
            "key": key,
            "score": 1.00
        }
    ]


def exact_search(text):
    parse_result = parse_text(text)
    query_result = query(parse_result)
    return query_result



