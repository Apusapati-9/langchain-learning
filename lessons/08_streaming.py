"""
Lesson 8: Streaming
--------------------
Covers:
  - Token streaming with LCEL (.stream())
  - Streaming with async (.astream())
  - Streaming intermediate steps in LangGraph
  - Stream events with .astream_events()
"""

import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from typing import Annotated, TypedDict


# ===========================================================================
# 8a. Basic token streaming with .stream()
# ===========================================================================

def demo_basic_streaming(llm: ChatOpenAI) -> None:
    print("\n--- 8a. Basic token streaming (.stream()) ---")
    chain = (
        ChatPromptTemplate.from_template("Write a 3-sentence story about a robot learning to cook.")
        | llm
        | StrOutputParser()
    )
    print("Streaming output: ", end="", flush=True)
    for chunk in chain.stream({}):
        print(chunk, end="", flush=True)
    print()  # newline after stream ends


# ===========================================================================
# 8b. Streaming with input variables
# ===========================================================================

def demo_streaming_with_input(llm: ChatOpenAI) -> None:
    print("\n--- 8b. Streaming with input variables ---")
    chain = (
        ChatPromptTemplate.from_template("Explain {concept} in exactly 2 sentences.")
        | llm
        | StrOutputParser()
    )
    topics = ["recursion", "closures"]
    for topic in topics:
        print(f"\nStreaming '{topic}': ", end="", flush=True)
        for chunk in chain.stream({"concept": topic}):
            print(chunk, end="", flush=True)
    print()


# ===========================================================================
# 8c. Async streaming with .astream()
# ===========================================================================

async def async_stream(llm: ChatOpenAI) -> None:
    print("\n--- 8c. Async streaming (.astream()) ---")
    chain = (
        ChatPromptTemplate.from_template("List 3 benefits of async programming in Python.")
        | llm
        | StrOutputParser()
    )
    print("Async stream: ", end="", flush=True)
    async for chunk in chain.astream({}):
        print(chunk, end="", flush=True)
    print()


def demo_async_streaming(llm: ChatOpenAI) -> None:
    asyncio.run(async_stream(llm))


# ===========================================================================
# 8d. Streaming intermediate steps in LangGraph
# ===========================================================================

def demo_langgraph_streaming(llm: ChatOpenAI) -> None:
    print("\n--- 8d. LangGraph node-level streaming (.stream()) ---")

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    def chat_node(state: State) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(State)
    graph.add_node("chat", chat_node)
    graph.set_entry_point("chat")
    graph.add_edge("chat", END)
    app = graph.compile()

    print("Streaming graph updates:")
    for update in app.stream(
        {"messages": [HumanMessage(content="Name 3 Python web frameworks in one line.")]},
        stream_mode="updates",
    ):
        for node, value in update.items():
            content = value["messages"][-1].content
            print(f"  [{node}] → {content}")


# ===========================================================================
# 8e. Streaming events with .astream_events()
# ===========================================================================

async def async_stream_events(llm: ChatOpenAI) -> None:
    print("\n--- 8e. Stream events (.astream_events()) ---")
    chain = (
        ChatPromptTemplate.from_template("What is a Python decorator? One sentence.")
        | llm.with_config({"run_name": "decorator_llm"})
        | StrOutputParser()
    )

    event_types_seen = set()
    tokens = []

    async for event in chain.astream_events({}, version="v2"):
        kind = event["event"]
        event_types_seen.add(kind)
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"].content
            if chunk:
                tokens.append(chunk)

    print(f"  Event types seen : {sorted(event_types_seen)}")
    print(f"  Final answer     : {''.join(tokens)}")


def demo_stream_events(llm: ChatOpenAI) -> None:
    asyncio.run(async_stream_events(llm))


# ===========================================================================
# Entry point
# ===========================================================================

def run():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    demo_basic_streaming(llm)
    demo_streaming_with_input(llm)
    demo_async_streaming(llm)
    demo_langgraph_streaming(llm)
    demo_stream_events(llm)
