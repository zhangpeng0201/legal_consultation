'''
精确搜索模块：
    精确索引 (法名, 条目, 是否增补)   ->   行索引
'''


import json
import re
from config import  FILE_NAME, load_data


CN_NUM = {'零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
          '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
UNITS = {'十': 10, '百': 100, '千': 1000}

sec_pattern = r'^第([零一二三四五六七八九十百千]+)条'
sup_pattern = r'^第[零一二三四五六七八九十百千]+条之([零一二三四五六七八九十百千]+)'

# 把中文数字转为阿拉伯数字
def cn_to_int(s: str) -> int:
    if s.isdigit():
        return int(s)
    total, num = 0, 0
    for char in s:
        if char in CN_NUM:
            num = CN_NUM[char]
        elif char in UNITS:
            unit = UNITS[char]
            total += (num if num else 1) * unit
            num = 0
    return total + num

# 获取条文编号
def get_sec(s: str) -> int:
    m = re.search(sec_pattern, s)
    if not m:
        return -1
    return cn_to_int(m.group(1))


# 判断条文是否为增补条文
def get_sup(s: str) -> int:
    m = re.search(sup_pattern, s)
    if m:
        return cn_to_int(m.group(1))
    else:
        return 0

# 构建精确索引
def build_exact_index(raw) -> dict:
    law_to_idx = {}
    for index, law in enumerate(raw):
        n = get_sec(law['section'])
        if n == -1:
            continue
        key = (law['title'], n, get_sup(law['section']))
        if key in law_to_idx:
            raise ValueError(f"索引冲突 {key}: 行 {law_to_idx[key]} vs 行 {index}")
        law_to_idx[key] = index
    # print(f"构建精确索引完成，共 {len(law_to_idx)} 个条目")
    return law_to_idx


def law_to_idx():
    raw = load_data(FILE_NAME)
    return build_exact_index(raw)



















