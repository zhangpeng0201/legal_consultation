import time
import uuid

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm import llm          # 复用现成实例，不新建连接

SESSIONS = {}          # {sid: {"history": [...], "at": 时间戳}}
TTL = 1800             # 30 分钟不活跃回收（惰性清理）
MAX_STORED = 20        # 每条会话最多留 20 条消息 = 10 轮
MAX_TURNS = 3          # 喂给改写模型的轮数
MAX_CHARS = 300        # 每条历史截断长度

ROLE_NAMES = {"user": "用户", "assistant": "助手"}


REWRITE_PROMPT = ChatPromptTemplate.from_template(
    '''
    【任务】把「用户最新一句话」改写成不依赖上文、可以独立检索的完整问句。

    【规则】
    1. 只输出改写后的那一句话，不要解释、不要引号、不要任何前缀
    2. 补全指代（"它""这个""他"）和省略，例如"那防卫过当呢" → "防卫过当的认定标准"
    3. 只做补全，不添加用户没问过的新内容
    4. 如果最新一句话本身已经完整，原样输出，一个字都不要改

    【对话历史】
    {history}

    【用户最新一句话】
    {query}

    输出：
    '''
)
REWRITE_CHAIN = REWRITE_PROMPT | llm | StrOutputParser()


def format_history(history):
    return "\n".join(f"{ROLE_NAMES.get(m['role'], m['role'])}：{m['content']}" for m in history)


def get_session(session_id):
    """返回 (history, sid)。传 None 或未知 id → 新建会话。"""
    now = time.time()
    for k in list(SESSIONS):                       # 惰性清理，list() 防运行时改字典
        if now - SESSIONS[k]["at"] > TTL:
            del SESSIONS[k]

    rec = SESSIONS.get(session_id)
    if rec is None:
        sid = uuid.uuid4().hex[:8]
        SESSIONS[sid] = {"history": [], "at": now}
        print(f"[session] new {sid}")
        return [], sid

    rec["at"] = now
    print(f"[session] {session_id} history={len(rec['history'])}")
    return rec["history"], session_id


def save_turn(sid, query, answer):
    rec = SESSIONS.get(sid)
    if rec is None:
        return
    rec["history"] += [
        {"role": "user", "content": query[:MAX_CHARS]},      # 存原句，不存改写句
        {"role": "assistant", "content": answer[:MAX_CHARS]},
    ]
    rec["history"] = rec["history"][-MAX_STORED:]             # 成对裁，保证从 user 开头
    rec["at"] = time.time()


def rewrite_query(query, history):
    if not history:                    # ← 单轮：一次 LLM 调用都不多
        return query
    try:
        text = REWRITE_CHAIN.invoke({
            "history": format_history(history[-MAX_TURNS * 2:]),
            "query": query,
        })
        text = text.strip().splitlines()[0].strip().strip('“”"\'')   # 甩掉啰嗦和引号
        if text and text != query:
            print(f"[改写] {query} -> {text}")
        return text or query
    except Exception as e:
        print(f"[改写失败，退回原句] {e}")
        return query


