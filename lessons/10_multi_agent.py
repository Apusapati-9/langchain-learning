"""
Lesson 10: Multi-Agent Systems
--------------------------------
Covers:
  - Supervisor pattern  — a router LLM delegates to specialist agents
  - Sequential handoffs — agents pass results to the next agent
  - Parallel agents     — run independent agents simultaneously, merge results
  - Shared state        — agents read/write a common state dict
"""

from typing import Annotated, TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
import operator


# ===========================================================================
# Shared tools
# ===========================================================================

@tool
def web_search(query: str) -> str:
    """Simulate a web search and return a mock result."""
    mock_results = {
        "python":     "Python 3.13 released Oct 2024. New features: free-threaded mode, improved JIT.",
        "langchain":  "LangChain 1.x released 2025. Removed AgentExecutor, now uses LangGraph.",
        "ai trends":  "Top AI trends 2025: multimodal models, agents, on-device inference, RAG.",
    }
    for key, result in mock_results.items():
        if key in query.lower():
            return result
    return f"Search results for '{query}': No specific mock data available."


@tool
def summarise_text(text: str) -> str:
    """Return a very short summary (first 100 chars) of the given text."""
    return text[:100].rstrip() + ("..." if len(text) > 100 else "")


@tool
def word_count(text: str) -> str:
    """Count words in text."""
    return f"{len(text.split())} words"


# ===========================================================================
# 10a. Supervisor pattern
# ===========================================================================

def demo_supervisor(llm: ChatOpenAI) -> None:
    print("\n--- 10a. Supervisor pattern ---")

    researcher = create_react_agent(
        llm, [web_search],
        prompt="You are a research agent. Use web_search to find information."
    )
    writer = create_react_agent(
        llm, [summarise_text, word_count],
        prompt="You are a writing agent. Summarise and analyse text."
    )

    class State(TypedDict):
        task: str
        next_agent: str
        research: str
        final: str

    def supervisor_node(state: State) -> dict:
        task = state["task"]
        if not state.get("research"):
            return {"next_agent": "researcher"}
        return {"next_agent": "writer"}

    def researcher_node(state: State) -> dict:
        result = researcher.invoke({"messages": [HumanMessage(content=state["task"])]})
        return {"research": result["messages"][-1].content}

    def writer_node(state: State) -> dict:
        msg = f"Summarise this research in one sentence:\n{state['research']}"
        result = writer.invoke({"messages": [HumanMessage(content=msg)]})
        return {"final": result["messages"][-1].content}

    def route(state: State) -> Literal["researcher", "writer", "__end__"]:
        if state["next_agent"] == "researcher":
            return "researcher"
        if state["next_agent"] == "writer":
            return "writer"
        return "__end__"

    graph = StateGraph(State)
    graph.add_node("supervisor",  supervisor_node)
    graph.add_node("researcher",  researcher_node)
    graph.add_node("writer",      writer_node)
    graph.set_entry_point("supervisor")
    graph.add_conditional_edges("supervisor", route)
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer",     END)

    app = graph.compile()
    result = app.invoke({"task": "Find information about Python and summarise it.", "next_agent": "", "research": "", "final": ""})
    print(f"  Research : {result['research'][:120]}...")
    print(f"  Final    : {result['final']}")


# ===========================================================================
# 10b. Sequential handoffs
# ===========================================================================

def demo_sequential_handoffs(llm: ChatOpenAI) -> None:
    print("\n--- 10b. Sequential handoffs (researcher → analyst → presenter) ---")

    class State(TypedDict):
        topic: str
        raw_info: str
        analysis: str
        presentation: str

    def researcher_node(state: State) -> dict:
        result = llm.invoke([
            SystemMessage(content="You are a researcher. Find 2 key facts."),
            HumanMessage(content=f"Research: {state['topic']}"),
        ])
        print("  [researcher] done")
        return {"raw_info": result.content}

    def analyst_node(state: State) -> dict:
        result = llm.invoke([
            SystemMessage(content="You are an analyst. Extract one key insight."),
            HumanMessage(content=f"Analyse this:\n{state['raw_info']}"),
        ])
        print("  [analyst] done")
        return {"analysis": result.content}

    def presenter_node(state: State) -> dict:
        result = llm.invoke([
            SystemMessage(content="You are a presenter. Write one punchy sentence."),
            HumanMessage(content=f"Present this insight:\n{state['analysis']}"),
        ])
        print("  [presenter] done")
        return {"presentation": result.content}

    graph = StateGraph(State)
    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst",    analyst_node)
    graph.add_node("presenter",  presenter_node)
    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst",    "presenter")
    graph.add_edge("presenter",  END)

    app = graph.compile()
    result = app.invoke({"topic": "benefits of open-source software", "raw_info": "", "analysis": "", "presentation": ""})
    print(f"\n  Presentation: {result['presentation']}")


# ===========================================================================
# 10c. Parallel agents with shared state
# ===========================================================================

def demo_parallel_agents(llm: ChatOpenAI) -> None:
    print("\n--- 10c. Parallel agents (fan-out → merge) ---")

    class State(TypedDict):
        topic: str
        pros: str
        cons: str
        verdict: str

    def pros_node(state: State) -> dict:
        result = llm.invoke([HumanMessage(content=f"List 2 pros of {state['topic']}. Be brief.")])
        return {"pros": result.content}

    def cons_node(state: State) -> dict:
        result = llm.invoke([HumanMessage(content=f"List 2 cons of {state['topic']}. Be brief.")])
        return {"cons": result.content}

    def merge_node(state: State) -> dict:
        result = llm.invoke([
            HumanMessage(content=(
                f"Given these pros and cons of '{state['topic']}', give a one-sentence verdict.\n"
                f"Pros: {state['pros']}\nCons: {state['cons']}"
            ))
        ])
        return {"verdict": result.content}

    graph = StateGraph(State)
    graph.add_node("pros",  pros_node)
    graph.add_node("cons",  cons_node)
    graph.add_node("merge", merge_node)

    # Fan-out from START to both pros and cons in parallel
    graph.set_entry_point("pros")
    graph.add_edge("__start__", "cons")
    graph.add_edge("pros",  "merge")
    graph.add_edge("cons",  "merge")
    graph.add_edge("merge", END)

    app = graph.compile()
    result = app.invoke({"topic": "microservices architecture", "pros": "", "cons": "", "verdict": ""})
    print(f"  Pros   : {result['pros'][:100]}...")
    print(f"  Cons   : {result['cons'][:100]}...")
    print(f"  Verdict: {result['verdict']}")


# ===========================================================================
# Entry point
# ===========================================================================

def run():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    demo_supervisor(llm)
    demo_sequential_handoffs(llm)
    demo_parallel_agents(llm)
