"""
Lesson 9: Document Loaders & Text Splitting
--------------------------------------------
Covers:
  - TextLoader       — plain .txt files
  - CSVLoader        — tabular data
  - DirectoryLoader  — load all files in a folder
  - RecursiveCharacterTextSplitter — general-purpose splitter
  - CharacterTextSplitter          — delimiter-based splitter
  - TokenTextSplitter              — split by token count
  - Inspecting chunk metadata
"""

import os
import csv
from pathlib import Path

from langchain_community.document_loaders import TextLoader, CSVLoader, DirectoryLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Sample data directory
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"


def create_sample_data() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    # Plain text file
    (DATA_DIR / "python_history.txt").write_text(
        "Python was created by Guido van Rossum and first released in 1991. "
        "It was designed to emphasize code readability, and its syntax allows "
        "programmers to express concepts in fewer lines of code than languages "
        "such as C++ or Java.\n\n"
        "Python 2.0 was released in 2000 and introduced features like list "
        "comprehensions and a garbage collection system. Python 3.0, released "
        "in 2008, was a major revision that was not backward-compatible with "
        "Python 2.\n\n"
        "Today Python is one of the most popular programming languages in the "
        "world, widely used in web development, data science, AI, and automation."
    )

    # CSV file
    with open(DATA_DIR / "languages.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["language", "year", "paradigm", "use_case"])
        writer.writeheader()
        writer.writerows([
            {"language": "Python",     "year": 1991, "paradigm": "multi-paradigm", "use_case": "AI, web, scripting"},
            {"language": "JavaScript", "year": 1995, "paradigm": "multi-paradigm", "use_case": "web frontend/backend"},
            {"language": "Rust",       "year": 2010, "paradigm": "multi-paradigm", "use_case": "systems, WebAssembly"},
            {"language": "Go",         "year": 2009, "paradigm": "concurrent",     "use_case": "cloud, CLI tools"},
            {"language": "TypeScript", "year": 2012, "paradigm": "multi-paradigm", "use_case": "large-scale web apps"},
        ])

    # Second text file (for directory loader)
    (DATA_DIR / "langchain_notes.txt").write_text(
        "LangChain is a framework for developing applications powered by LLMs. "
        "Key components include: prompts, chains, memory, agents, and retrievers.\n\n"
        "LCEL (LangChain Expression Language) uses the pipe operator to chain "
        "components together declaratively. It supports streaming, batching, "
        "and async execution natively."
    )

    print(f"  Sample data written to: {DATA_DIR}")


# ===========================================================================
# 9a. TextLoader
# ===========================================================================

def demo_text_loader() -> None:
    print("\n--- 9a. TextLoader ---")
    loader = TextLoader(str(DATA_DIR / "python_history.txt"))
    docs = loader.load()
    print(f"  Documents loaded : {len(docs)}")
    print(f"  Source           : {docs[0].metadata['source']}")
    print(f"  Content preview  : {docs[0].page_content[:120]}...")


# ===========================================================================
# 9b. CSVLoader
# ===========================================================================

def demo_csv_loader() -> None:
    print("\n--- 9b. CSVLoader ---")
    loader = CSVLoader(str(DATA_DIR / "languages.csv"))
    docs = loader.load()
    print(f"  Documents loaded : {len(docs)}  (one per row)")
    for doc in docs[:2]:
        print(f"  Row → {doc.page_content[:80]}")


# ===========================================================================
# 9c. DirectoryLoader
# ===========================================================================

def demo_directory_loader() -> None:
    print("\n--- 9c. DirectoryLoader (all .txt files) ---")
    loader = DirectoryLoader(str(DATA_DIR), glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
    print(f"  Files loaded : {len(docs)}")
    for doc in docs:
        src = Path(doc.metadata["source"]).name
        print(f"  {src} — {len(doc.page_content)} chars")


# ===========================================================================
# 9d. RecursiveCharacterTextSplitter
# ===========================================================================

def demo_recursive_splitter() -> None:
    print("\n--- 9d. RecursiveCharacterTextSplitter ---")
    loader = TextLoader(str(DATA_DIR / "python_history.txt"))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"  1 doc → {len(chunks)} chunks  (size≤200, overlap=30)")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1} [{len(chunk.page_content)} chars]: {chunk.page_content[:70]}...")


# ===========================================================================
# 9e. CharacterTextSplitter
# ===========================================================================

def demo_character_splitter() -> None:
    print("\n--- 9e. CharacterTextSplitter (split on paragraphs) ---")
    loader = TextLoader(str(DATA_DIR / "python_history.txt"))
    docs = loader.load()

    splitter = CharacterTextSplitter(separator="\n\n", chunk_size=500, chunk_overlap=0)
    chunks = splitter.split_documents(docs)
    print(f"  1 doc → {len(chunks)} paragraph chunks")
    for i, chunk in enumerate(chunks):
        print(f"  Paragraph {i+1}: {chunk.page_content[:80]}...")


# ===========================================================================
# 9f. TokenTextSplitter + metadata inspection
# ===========================================================================

def demo_token_splitter() -> None:
    print("\n--- 9f. TokenTextSplitter + metadata ---")
    loader = TextLoader(str(DATA_DIR / "langchain_notes.txt"))
    docs = loader.load()

    splitter = TokenTextSplitter(chunk_size=30, chunk_overlap=5)
    chunks = splitter.split_documents(docs)
    print(f"  1 doc → {len(chunks)} token-based chunks (≤30 tokens each)")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1} | source: {Path(chunk.metadata['source']).name}")
        print(f"           | text  : {chunk.page_content[:80]}")


# ===========================================================================
# Entry point
# ===========================================================================

def run():
    print("\nCreating sample data files...")
    create_sample_data()

    demo_text_loader()
    demo_csv_loader()
    demo_directory_loader()
    demo_recursive_splitter()
    demo_character_splitter()
    demo_token_splitter()
