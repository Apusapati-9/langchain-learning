"""
Lesson 14: LangChain + FastAPI
--------------------------------
Covers:
  - Wrapping a LangChain chain in a FastAPI endpoint
  - Streaming responses via Server-Sent Events
  - Chat endpoint with per-session memory
  - Testing endpoints with httpx

Note: This lesson starts a real HTTP server on port 8765,
      runs automated tests against it, then shuts it down.
"""

import asyncio
import threading
import time
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


PORT = 8765
BASE_URL = f"http://127.0.0.1:{PORT}"

# ---------------------------------------------------------------------------
# Shared LLM & chains (initialised once at server startup)
# ---------------------------------------------------------------------------

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# Simple Q&A chain
qa_chain = (
    ChatPromptTemplate.from_template("Answer concisely: {question}")
    | llm
    | StrOutputParser()
)

# Chat chain with memory
_store: dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Be concise."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])
chat_chain = RunnableWithMessageHistory(
    chat_prompt | llm | StrOutputParser(),
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="LangChain API", version="1.0")


class QuestionRequest(BaseModel):
    question: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: QuestionRequest):
    """Simple Q&A — returns a JSON response."""
    answer = qa_chain.invoke({"question": req.question})
    return {"question": req.question, "answer": answer}


@app.post("/stream")
def stream(req: QuestionRequest):
    """Streaming Q&A — returns tokens via Server-Sent Events."""
    def token_generator():
        for chunk in qa_chain.stream({"question": req.question}):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


@app.post("/chat")
def chat(req: ChatRequest):
    """Stateful chat — memory keyed by session_id."""
    response = chat_chain.invoke(
        {"input": req.message},
        config={"configurable": {"session_id": req.session_id}},
    )
    return {"session_id": req.session_id, "response": response}


# ---------------------------------------------------------------------------
# Server lifecycle helpers
# ---------------------------------------------------------------------------

def start_server() -> threading.Thread:
    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # Wait until the server is ready
    for _ in range(30):
        try:
            httpx.get(f"{BASE_URL}/health", timeout=1)
            break
        except Exception:
            time.sleep(0.3)
    return thread


# ---------------------------------------------------------------------------
# Demo tests
# ---------------------------------------------------------------------------

def demo_health(client: httpx.Client) -> None:
    print("\n--- 14a. GET /health ---")
    r = client.get("/health")
    print(f"  Status : {r.status_code}")
    print(f"  Body   : {r.json()}")


def demo_ask(client: httpx.Client) -> None:
    print("\n--- 14b. POST /ask ---")
    r = client.post("/ask", json={"question": "What is a Python generator?"})
    data = r.json()
    print(f"  Question : {data['question']}")
    print(f"  Answer   : {data['answer'][:100]}...")


def demo_stream(client: httpx.Client) -> None:
    print("\n--- 14c. POST /stream (Server-Sent Events) ---")
    tokens = []
    with client.stream("POST", "/stream", json={"question": "Name 3 Python web frameworks."}) as r:
        for line in r.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                tokens.append(line[6:])
    print(f"  Streamed answer: {''.join(tokens)[:100]}...")
    print(f"  Total chunks   : {len(tokens)}")


def demo_chat_memory(client: httpx.Client) -> None:
    print("\n--- 14d. POST /chat (memory across turns) ---")
    session = "lesson-14-demo"
    turns = [
        "My name is Alice and I love Python.",
        "What is my name and favourite language?",
    ]
    for msg in turns:
        r = client.post("/chat", json={"session_id": session, "message": msg})
        data = r.json()
        print(f"\n  User : {msg}")
        print(f"  Bot  : {data['response'][:100]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    print("\nStarting FastAPI server on port", PORT)
    start_server()
    print("Server ready.\n")

    with httpx.Client(base_url=BASE_URL, timeout=30) as client:
        demo_health(client)
        demo_ask(client)
        demo_stream(client)
        demo_chat_memory(client)

    print("\n[Server running in background — will stop when lesson ends]")
