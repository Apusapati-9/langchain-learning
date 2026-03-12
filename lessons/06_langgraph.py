"""
Lesson 6: LangGraph
--------------------
Covers:
  - Core concepts: StateGraph, nodes, edges
  - Conditional edges (branching)
  - Built-in message state with add_messages
  - Multi-node LLM pipeline (plan → execute → review)
  - ReAct agent built from scratch using LangGraph
"""

import operator
from typing import Annotated, TypedDict, Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# ===========================================================================
# 6a. Minimal graph — two nodes, one edge
# ===========================================================================

def demo_minimal_graph() -> None:
    print("\n--- 6a. Minimal graph (node → node → END) ---")

    class State(TypedDict):
        text: str
        result: str

    def shout_node(state: State) -> dict:
        return {"text": state["text"].upper()}

    def exclaim_node(state: State) -> dict:
        return {"result": state["text"] + "!!!"}

    graph = StateGraph(State)
    graph.add_node("shout",   shout_node)
    graph.add_node("exclaim", exclaim_node)
    graph.set_entry_point("shout")
    graph.add_edge("shout", "exclaim")
    graph.add_edge("exclaim", END)

    app = graph.compile()
    output = app.invoke({"text": "hello langgraph", "result": ""})
    print("Input : 'hello langgraph'")
    print("Output:", output["result"])


# ===========================================================================
# 6b. Conditional edges — route based on state
# ===========================================================================

def demo_conditional_edges() -> None:
    print("\n--- 6b. Conditional edges (branching) ---")

    class State(TypedDict):
        number: int
        verdict: str

    def classify(state: State) -> Literal["even_node", "odd_node"]:
        return "even_node" if state["number"] % 2 == 0 else "odd_node"

    def even_node(state: State) -> dict:
        return {"verdict": f"{state['number']} is EVEN"}

    def odd_node(state: State) -> dict:
        return {"verdict": f"{state['number']} is ODD"}

    graph = StateGraph(State)
    graph.add_node("even_node", even_node)
    graph.add_node("odd_node",  odd_node)
    graph.set_entry_point("router")
    graph.add_node("router", lambda s: s)          # pass-through router node
    graph.add_conditional_edges("router", classify)
    graph.add_edge("even_node", END)
    graph.add_edge("odd_node",  END)

    app = graph.compile()
    for n in [4, 7]:
        out = app.invoke({"number": n, "verdict": ""})
        print(f"  {n} →", out["verdict"])


# ===========================================================================
# 6c. Multi-node LLM pipeline: plan → draft → critique
# ===========================================================================

def demo_llm_pipeline(llm: ChatOpenAI) -> None:
    print("\n--- 6c. Multi-node LLM pipeline (plan → draft → critique) ---")

    class State(TypedDict):
        topic: str
        plan: str
        draft: str
        critique: str

    def plan_node(state: State) -> dict:
        prompt = f"List 3 bullet-point ideas for a short blog post about: {state['topic']}"
        plan = llm.invoke([HumanMessage(content=prompt)]).content
        print(f"\n  [plan_node] done")
        return {"plan": plan}

    def draft_node(state: State) -> dict:
        prompt = (
            f"Write a short 3-sentence blog intro about '{state['topic']}' "
            f"using this outline:\n{state['plan']}"
        )
        draft = llm.invoke([HumanMessage(content=prompt)]).content
        print(f"  [draft_node] done")
        return {"draft": draft}

    def critique_node(state: State) -> dict:
        prompt = (
            f"Give one sentence of constructive feedback on this blog intro:\n{state['draft']}"
        )
        critique = llm.invoke([HumanMessage(content=prompt)]).content
        print(f"  [critique_node] done")
        return {"critique": critique}

    graph = StateGraph(State)
    graph.add_node("plan",     plan_node)
    graph.add_node("draft",    draft_node)
    graph.add_node("critique", critique_node)
    graph.set_entry_point("plan")
    graph.add_edge("plan",     "draft")
    graph.add_edge("draft",    "critique")
    graph.add_edge("critique", END)

    app = graph.compile()
    result = app.invoke({"topic": "why Python is great for beginners", "plan": "", "draft": "", "critique": ""})

    print(f"\n  Topic   : {result['topic']}")
    print(f"\n  Plan    :\n{result['plan']}")
    print(f"\n  Draft   :\n{result['draft']}")
    print(f"\n  Critique: {result['critique']}")


# ===========================================================================
# 6d. Message-state graph with add_messages reducer
# ===========================================================================

def demo_message_state(llm: ChatOpenAI) -> None:
    print("\n--- 6d. Message state graph (add_messages reducer) ---")

    class State(TypedDict):
        # add_messages merges new messages into the list instead of replacing it
        messages: Annotated[list, add_messages]

    system_msg = SystemMessage(content="You are a concise assistant. Reply in one sentence.")

    def chat_node(state: State) -> dict:
        response = llm.invoke([system_msg] + state["messages"])
        return {"messages": [response]}

    graph = StateGraph(State)
    graph.add_node("chat", chat_node)
    graph.set_entry_point("chat")
    graph.add_edge("chat", END)
    app = graph.compile()

    # Simulate a two-turn conversation by passing accumulated messages
    turns = [
        "What is a Python generator?",
        "Give me a one-line code example.",
    ]
    state = {"messages": []}
    for user_input in turns:
        state["messages"].append(HumanMessage(content=user_input))
        state = app.invoke(state)
        print(f"\n  User: {user_input}")
        print(f"  Bot : {state['messages'][-1].content}")


# ===========================================================================
# 6e. ReAct agent from scratch with ToolNode
# ===========================================================================

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def demo_scratch_agent(llm: ChatOpenAI) -> None:
    print("\n--- 6e. ReAct agent from scratch with ToolNode ---")

    tools = [multiply, add]
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    def agent_node(state: State) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: State) -> Literal["tools", "__end__"]:
        last = state["messages"][-1]
        # If the LLM returned tool calls, route to tool_node; otherwise finish
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "__end__"

    graph = StateGraph(State)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")   # loop back after tool execution

    app = graph.compile()

    query = "What is (7 + 3) multiplied by 4?"
    print(f"\n  Query: {query}")
    result = app.invoke({"messages": [HumanMessage(content=query)]})
    print(f"  Answer: {result['messages'][-1].content}")

    print("\n  Graph flow: agent → tools → agent → END")
    print("  (agent calls add → loops back → calls multiply → loops back → final answer)")


# ===========================================================================
# Entry point
# ===========================================================================

def run():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    demo_minimal_graph()
    demo_conditional_edges()
    demo_llm_pipeline(llm)
    demo_message_state(llm)
    demo_scratch_agent(llm)
