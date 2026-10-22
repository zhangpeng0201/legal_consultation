import os
from dotenv import load_dotenv
from config import FILE_NAME, load_data
from rrf_search import rrf_search
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

load_dotenv()

DATA = load_data(FILE_NAME)

llm = ChatOpenAI(
    api_key=os.environ['DEEPSEEK_API_KEY'],
    model_name=os.environ['DEEPSEEK_NAME'],
    base_url=os.environ['DEEPSEEK_URL'],
    temperature=0,
    max_tokens=800,
    extra_body={"thinking": {"type": "disabled"}},
)

JUDGE_PROMPT = ChatPromptTemplate.from_template(
    '''
    你是一个法律问题判断专家。你的任务是判断：给定的参考法律条文是否足以回答该法律问题，并以 JSON 格式输出判断结果。
    
    【最高优先级规则，优先于下面所有判断标准】
    规则一 以【参考法律条文】为准。禁止用你记忆中的法律知识判断条文内容是否正确——法律可能已修订，即使该条内容与你记忆中的版本不一致，也一律视为有效条文。
    规则二 只要【参考法律条文】中出现了用户所问条号对应的原文，answerable 必须为 true。

    【判断标准】（按顺序逐条检查）
    1. 条文中必须存在能直接支撑回答的核心条文——你能明确指出"第几条"回答了问题的争议点
    2. 若问题本身只是对某条文的调用（例如"公司法第23条""民法典第五百八十五条"），属于"查询该条内容"，只要参考条文中包含该条原文，即 answerable 为 true
    3. 仅出现相关主题的字眼（如都提到"合同""借款"）但条文内容与问题的争议点不对应 → answerable 为 false
    4. 条文只能部分覆盖问题（如只涉及责任认定、不涉及用户问的赔偿标准）→ answerable 为 false
    5. 条文之间互相矛盾、或需要条文以外的专业知识才能得出结论 → answerable 为 false
    6. 拿不准时 answerable 一律为 false

    【输出格式】
    只输出一个 JSON 对象，不输出任何解释、前缀、代码块标记（```）或 JSON 以外的任何内容。字段定义：
    - answerable：布尔值，true 表示条文足以回答，false 表示不足以回答
    - evidence：字符串数组，列出支撑判断的法律全称 + 条文序号（如"中华人民共和国民法典 第一千二百五十四条"）；answerable 为 false 时输出空数组 []
    - reason：一句话说明判断依据；answerable 为 false 时须简述具体缺什么（如"条文仅涉及 X，未涉及用户所问的 Y"）

    输出示例：
    {{"answerable": true, "evidence": ["中华人民共和国民法典 第一千二百五十四条"], "reason": "该条直接规定高空抛物由侵权人担责，难以确定时由可能加害的建筑物使用人补偿，与问题争议点直接对应"}}

    【待判断问题】
    {query}

    【参考法律条文】
    {context}

    输出：
    '''
)
JUDGE_CHAIN = JUDGE_PROMPT | llm | JsonOutputParser()

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    '''
    【回答要求】
    1. 严格依据【参考法律条文】回答，禁止引用条文以外的任何法律依据；条文未覆盖的部分须明确说明"提供的条文中未涉及该问题"
    2. 只引用你实际用到的条文，每条条文引用一次，不重复引用；不臆造、不脑补不存在的法条或条文内容
    3. 总字数（不含条文原文）不超过 200 字

    【结构要求】
    使用 Markdown 格式：
    1. **问题结论**：1-2 句话直接回答，句末标注（法律全称 第X条）
    2. **法律依据**：只列用到的条文——法律全称 + 条文序号 + 条文原文（引用格式），不写分析
    3. **简要分析**：1-2 句话说明条文如何对应问题，没有额外分析就写"该条直接对应"

    【禁止】
    - 同一条条文在结论和依据两段重复出现
    - 输出与问题无关的条文，哪怕它在参考条文中
    - 空话套话（"根据相关法律规定""综上所述建议咨询律师"之类一律不写）

    【问题】
    {query}

    【参考法律条文】
    {context}
    '''
)
ANSWER_CHAIN = ANSWER_PROMPT | llm | StrOutputParser()


# 获取查询结果
def get_search_result(query):
    row_result = []
    search_result = rrf_search(query)
    for result in search_result:
        row = DATA[result['lineno']]
        row_result.append(row['title'] + ' ' + row['section'])
    context = "\n".join(f"[{j}] {t}" for j, t in enumerate(row_result, 1))
    return context


# 判断问题是否可以回答
def query_judge(query, context):
    return JUDGE_CHAIN.invoke({"query": query, "context": context})


# 答案生成
def generate_answer(query, context):
    return ANSWER_CHAIN.invoke({"query": query, "context": context})


async def generate_answer_stream(query, context):
    async for chunk in ANSWER_CHAIN.astream({"query": query, "context": context}):
        if not chunk:
            continue
        yield chunk


# 对话
def ask(query, context, judge):
    if not judge['answerable']:
        return {
            "answerable": False,
            "evidence": [],
            "reason": judge['reason'],
            "answer": None,
        }
    return {
        "answerable": True,
        "evidence": judge['evidence'],
        "reason": judge['reason'],
        "answer": generate_answer(query, context),
    }


# 对话（流式）
async def ask_stream(query, context, judge):
    if not judge['answerable']:
        yield {
            "type": "meta",
            "query": query,
            "answerable": False,
            "evidence": [],
            "reason": judge['reason'],
        }
        yield {"type": "done"}
        return
    yield {
        "type": "meta",
        "query": query,
        "answerable": True,
        "evidence": judge['evidence'],
        "reason": judge['reason'],
    }
    yield {"type": "status", "stage": "generating", "text": "正在起草答案"}
    try:
        async for chunk in generate_answer_stream(query, context):
            yield {"type": "delta", "text": chunk}
    except Exception as e:
        yield {"type": "error", "message": str(e)}
    yield {"type": "done"}
