"""
Lesson 13: Callbacks & Tracing
--------------------------------
Covers:
  - BaseCallbackHandler — custom event hooks
  - Token usage tracking
  - Timing callback (latency per chain step)
  - Logging callback (structured execution trace)
  - Chaining multiple callbacks together
"""

import time
from typing import Any
from uuid import UUID

from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.outputs import LLMResult
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage


# ===========================================================================
# Custom callback handlers
# ===========================================================================

class TokenUsageCallback(BaseCallbackHandler):
    """Tracks total prompt and completion tokens across all LLM calls."""

    def __init__(self):
        self.prompt_tokens     = 0
        self.completion_tokens = 0
        self.llm_calls         = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            for gen in generations:
                usage = getattr(gen.message, "usage_metadata", None) or {}
                self.prompt_tokens     += usage.get("input_tokens", 0)
                self.completion_tokens += usage.get("output_tokens", 0)
        self.llm_calls += 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def report(self) -> None:
        print(f"  LLM calls        : {self.llm_calls}")
        print(f"  Prompt tokens    : {self.prompt_tokens}")
        print(f"  Completion tokens: {self.completion_tokens}")
        print(f"  Total tokens     : {self.total_tokens}")


class TimingCallback(BaseCallbackHandler):
    """Records wall-clock time for each LLM call."""

    def __init__(self):
        self._start: dict[str, float] = {}
        self.timings: list[float] = []

    def on_llm_start(self, serialized: dict, messages: list, *, run_id: UUID, **kwargs: Any) -> None:
        self._start[str(run_id)] = time.perf_counter()

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        elapsed = time.perf_counter() - self._start.pop(str(run_id), time.perf_counter())
        self.timings.append(elapsed)
        print(f"    [TimingCallback] LLM call finished in {elapsed:.2f}s")


class LoggingCallback(BaseCallbackHandler):
    """Prints a structured trace of chain events."""

    def on_chain_start(self, serialized: dict, inputs: dict, *, run_id: UUID, **kwargs: Any) -> None:
        name = (serialized or {}).get("name", "Chain")
        keys = list(inputs.keys()) if isinstance(inputs, dict) else type(inputs).__name__
        print(f"    [LoggingCallback] >> {name} started | inputs: {keys}")

    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        keys = list(outputs.keys()) if isinstance(outputs, dict) else type(outputs).__name__
        print(f"    [LoggingCallback] << Chain ended   | outputs: {keys}")

    def on_llm_start(self, serialized: dict, messages: list, *, run_id: UUID, **kwargs: Any) -> None:
        print(f"    [LoggingCallback] ~~ LLM call starting...")

    def on_llm_error(self, error: Exception, *, run_id: UUID, **kwargs: Any) -> None:
        print(f"    [LoggingCallback] !! LLM error: {error}")


# ===========================================================================
# 13a. Token usage tracking
# ===========================================================================

def demo_token_tracking(llm: ChatOpenAI) -> None:
    print("\n--- 13a. Token usage tracking ---")
    tracker = TokenUsageCallback()

    chain = (
        ChatPromptTemplate.from_template("Explain {topic} in one sentence.")
        | llm.with_config({"callbacks": [tracker]})
        | StrOutputParser()
    )

    topics = ["recursion", "closures", "decorators"]
    for topic in topics:
        result = chain.invoke({"topic": topic})
        print(f"  {topic}: {result[:70]}...")

    print()
    tracker.report()


# ===========================================================================
# 13b. Timing callback
# ===========================================================================

def demo_timing(llm: ChatOpenAI) -> None:
    print("\n--- 13b. Timing callback ---")
    timer = TimingCallback()

    chain = (
        ChatPromptTemplate.from_template("Name 3 uses of {language} programming language.")
        | llm.with_config({"callbacks": [timer]})
        | StrOutputParser()
    )

    for lang in ["Python", "JavaScript"]:
        print(f"\n  Calling for '{lang}':")
        chain.invoke({"language": lang})

    print(f"\n  Timings: {[f'{t:.2f}s' for t in timer.timings]}")
    print(f"  Average: {sum(timer.timings)/len(timer.timings):.2f}s")


# ===========================================================================
# 13c. Logging callback — structured trace
# ===========================================================================

def demo_logging_callback(llm: ChatOpenAI) -> None:
    print("\n--- 13c. Logging callback (execution trace) ---")
    logger = LoggingCallback()

    chain = (
        ChatPromptTemplate.from_template("What is {concept}? One sentence.")
        | llm
        | StrOutputParser()
    )

    result = chain.invoke(
        {"concept": "memoization"},
        config={"callbacks": [logger]},
    )
    print(f"\n  Answer: {result}")


# ===========================================================================
# 13d. Multiple callbacks together
# ===========================================================================

def demo_multi_callback(llm: ChatOpenAI) -> None:
    print("\n--- 13d. Multiple callbacks together ---")
    tracker = TokenUsageCallback()
    timer   = TimingCallback()

    chain = (
        ChatPromptTemplate.from_template("Give a one-sentence definition of {term}.")
        | llm.with_config({"callbacks": [tracker, timer]})
        | StrOutputParser()
    )

    terms = ["polymorphism", "encapsulation", "abstraction"]
    for term in terms:
        result = chain.invoke({"term": term})
        print(f"  {term}: {result[:70]}...")

    print(f"\n  Total tokens used : {tracker.total_tokens}")
    print(f"  Total time        : {sum(timer.timings):.2f}s")
    print(f"  Avg time/call     : {sum(timer.timings)/len(timer.timings):.2f}s")


# ===========================================================================
# Entry point
# ===========================================================================

def run():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    demo_token_tracking(llm)
    demo_timing(llm)
    demo_logging_callback(llm)
    demo_multi_callback(llm)
