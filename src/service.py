import json
import time
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from config import BASE_DIR
from llm import ask, query_judge, get_search_result, ask_stream
from contextlib import asynccontextmanager
from rrf_search import rrf_search
from session import get_session, save_turn, rewrite_query



@asynccontextmanager
async def lifespan(app: FastAPI):
    rrf_search('预热')   # 这一行会触发：bm25 pickle + vecs.npy + keys + bge 模型
    yield


app = FastAPI(title="法律条文查询", lifespan=lifespan)

# 前端页面：同源提供，免去跨域配置
app.mount("/html", StaticFiles(directory=BASE_DIR / "html", html=True), name="html")

@app.get("/")
def index():
    return RedirectResponse("/html/index.html")

class AskRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


def sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

@app.post("/ask/stream")
async def ask_stream_route(req: AskRequest):
    t0 = time.time()
    history, sid = get_session(req.session_id)
    q = rewrite_query(req.query, history)
    
    context = get_search_result(q)
    t1 = time.time()
    judge = query_judge(q, context)
    t2 = time.time()

    async def event_gen():
        yield sse({'type': 'session', 'session_id': sid})
        answer_text = ''
        ttfb = None
        async for event in ask_stream(q, context, judge):
            t = event.get('type')
            if t == 'delta':
                answer_text += event['text']
            elif t == 'meta' and not event.get('answerable'):
                answer_text = f"（条文中无依据，未作答）{event.get('reason', '')}"
            if ttfb is None:
                ttfb = time.time() - t0
            yield sse(event)
        save_turn(sid, req.query, answer_text)
        print(f"检索 {t1 - t0:.2f}s | judge {t2 - t1:.2f}s | TTFB {ttfb:.2f}s | 总计 {time.time() - t0:.2f}s")

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.post("/ask")
def ask_route(req: AskRequest):
    t0 = time.time()
    context = get_search_result(req.query)
    t1 = time.time()
    judge = query_judge(req.query, context)
    t2 = time.time()
    result = ask(req.query, context, judge)
    t3 = time.time()
    print(f"检索 {t1 - t0:.2f}s | judge {t2 - t1:.2f}s | generate {t3 - t2:.2f}s")
    return {"query": req.query, **result}
