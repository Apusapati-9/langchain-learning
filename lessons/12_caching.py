"""
Lesson 12: Caching
-------------------
Covers:
  - InMemoryCache  — avoid duplicate LLM calls within a session
  - SQLiteCache    — persist cache across sessions
  - Measuring latency & cost savings with caching
  - Semantic caching concept (exact-match vs meaning-based)
"""

import time
import os
from pathlib import Path

import langchain
from langchain_openai import ChatOpenAI
from langchain_core.caches import InMemoryCache
from langchain_community.cache import SQLiteCache
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.globals import set_llm_cache


CACHE_DB_PATH = Path(__file__).parent.parent / "data" / "llm_cache.db"


# ===========================================================================
# 12a. No cache baseline
# ===========================================================================

def demo_no_cache(llm: ChatOpenAI) -> None:
    print("\n--- 12a. Baseline — no cache ---")
    set_llm_cache(None)

    chain = (
        ChatPromptTemplate.from_template("What is the capital of {country}? One word.")
        | llm
        | StrOutputParser()
    )

    countries = ["France", "France"]   # same query twice
    for country in countries:
        t0 = time.perf_counter()
        result = chain.invoke({"country": country})
        elapsed = time.perf_counter() - t0
        print(f"  {country}: '{result.strip()}' in {elapsed:.2f}s")


# ===========================================================================
# 12b. InMemoryCache
# ===========================================================================

def demo_in_memory_cache(llm: ChatOpenAI) -> None:
    print("\n--- 12b. InMemoryCache (same-session deduplication) ---")
    set_llm_cache(InMemoryCache())

    chain = (
        ChatPromptTemplate.from_template("What is the capital of {country}? One word.")
        | llm
        | StrOutputParser()
    )

    countries = ["Germany", "Germany", "Japan", "Japan"]
    for country in countries:
        t0 = time.perf_counter()
        result = chain.invoke({"country": country})
        elapsed = time.perf_counter() - t0
        cached = elapsed < 0.05        # cache hits are nearly instant
        print(f"  {country}: '{result.strip()}' in {elapsed:.3f}s {'[CACHE HIT]' if cached else '[API CALL]'}")

    set_llm_cache(None)


# ===========================================================================
# 12c. SQLiteCache — persistent across sessions
# ===========================================================================

def demo_sqlite_cache(llm: ChatOpenAI) -> None:
    print("\n--- 12c. SQLiteCache (persisted across sessions) ---")
    CACHE_DB_PATH.parent.mkdir(exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=str(CACHE_DB_PATH)))

    chain = (
        ChatPromptTemplate.from_template("Name the inventor of {thing}. One sentence.")
        | llm
        | StrOutputParser()
    )

    queries = ["the telephone", "the telephone", "the light bulb"]
    for query in queries:
        t0 = time.perf_counter()
        result = chain.invoke({"thing": query})
        elapsed = time.perf_counter() - t0
        cached = elapsed < 0.05
        print(f"  '{query}': {result.strip()[:60]}... ({elapsed:.3f}s) {'[CACHE HIT]' if cached else '[API CALL]'}")

    print(f"\n  Cache DB saved to: {CACHE_DB_PATH}")
    set_llm_cache(None)


# ===========================================================================
# 12d. Cache invalidation & cost awareness
# ===========================================================================

def demo_cache_stats(llm: ChatOpenAI) -> None:
    print("\n--- 12d. Cache benefit summary ---")
    cache = InMemoryCache()
    set_llm_cache(cache)

    chain = (
        ChatPromptTemplate.from_template("Give a one-word synonym for {word}.")
        | llm
        | StrOutputParser()
    )

    words = ["happy", "sad", "fast", "happy", "sad", "fast", "happy"]
    api_calls = 0
    cache_hits = 0

    for word in words:
        t0 = time.perf_counter()
        chain.invoke({"word": word})
        elapsed = time.perf_counter() - t0
        if elapsed < 0.05:
            cache_hits += 1
        else:
            api_calls += 1

    total = len(words)
    print(f"  Total queries : {total}")
    print(f"  API calls     : {api_calls}  (billed)")
    print(f"  Cache hits    : {cache_hits}  (free)")
    print(f"  Savings       : {cache_hits/total*100:.0f}% of calls avoided")

    set_llm_cache(None)


# ===========================================================================
# Entry point
# ===========================================================================

def run():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    demo_no_cache(llm)
    demo_in_memory_cache(llm)
    demo_sqlite_cache(llm)
    demo_cache_stats(llm)
