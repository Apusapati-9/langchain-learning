"""
Lesson 4: Agents & Tools
--------------------------
Covers:
  - Defining custom tools with @tool
  - Creating a ReAct agent (create_react_agent)
  - AgentExecutor with verbose output
  - Tool with structured (multi-arg) input
"""

import math
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@tool
def get_current_datetime() -> str:
    """Return the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a safe mathematical expression and return the result.
    Supports: +, -, *, /, **, sqrt(), pi, e, sin(), cos(), tan(), log().
    Example: '2 ** 10', 'sqrt(144)', 'pi * 5 ** 2'
    """
    safe_env = {
        "sqrt": math.sqrt, "pi": math.pi, "e": math.e,
        "sin": math.sin,   "cos": math.cos, "tan": math.tan,
        "log": math.log,   "abs": abs,      "round": round,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, safe_env)  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error evaluating '{expression}': {exc}"


@tool
def word_counter(text: str) -> str:
    """Count the number of words and characters in the given text."""
    words = len(text.split())
    chars = len(text)
    return f"{words} words, {chars} characters"


@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert a value between common units.
    Supported conversions:
      Temperature : celsius ↔ fahrenheit, celsius ↔ kelvin
      Distance    : km ↔ miles, meters ↔ feet
    """
    key = (from_unit.lower(), to_unit.lower())
    conversions: dict[tuple[str, str], float | None] = {
        ("celsius", "fahrenheit"): None,
        ("fahrenheit", "celsius"): None,
        ("celsius", "kelvin"):     None,
        ("kelvin", "celsius"):     None,
        ("km", "miles"):           0.621371,
        ("miles", "km"):           1.60934,
        ("meters", "feet"):        3.28084,
        ("feet", "meters"):        0.3048,
    }
    if key not in conversions:
        return f"Unsupported conversion: {from_unit} → {to_unit}"

    if key == ("celsius", "fahrenheit"):
        return f"{value * 9/5 + 32:.2f} °F"
    if key == ("fahrenheit", "celsius"):
        return f"{(value - 32) * 5/9:.2f} °C"
    if key == ("celsius", "kelvin"):
        return f"{value + 273.15:.2f} K"
    if key == ("kelvin", "celsius"):
        return f"{value - 273.15:.2f} °C"

    factor = conversions[key]
    return f"{value * factor:.4f} {to_unit}"


# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------

def build_agent(llm: ChatOpenAI):
    tools = [get_current_datetime, calculator, word_counter, unit_converter]
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=(
            "You are a smart assistant with access to tools. "
            "Use them whenever they can help answer the user's question accurately."
        ),
    )


def invoke_agent(agent, query: str) -> str:
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content


# ---------------------------------------------------------------------------
# Demo runs
# ---------------------------------------------------------------------------

def run():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = build_agent(llm)

    queries = [
        "What is today's date and time?",
        "What is the square root of 1764, and then multiply that result by 7?",
        "Count the words in: 'LangChain makes building LLM apps much easier.'",
        "Convert 100 Celsius to Fahrenheit, and also convert 26.2 miles to km.",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("="*60)
        answer = invoke_agent(agent, query)
        print(f"Answer: {answer}")
