# LangChain Learning App

An interactive CLI app to learn core LangChain concepts through hands-on examples.

## Lessons

| # | Topic | Concepts Covered |
|---|-------|-----------------|
| 1 | LLMs & Prompt Templates | Direct LLM invocation, `PromptTemplate`, `ChatPromptTemplate`, few-shot prompting |
| 2 | Chains (LCEL) | `\|` pipe operator, `StrOutputParser`, `RunnableParallel`, `RunnablePassthrough` |
| 3 | Memory & Chat History | `ChatMessageHistory`, `RunnableWithMessageHistory`, multi-session isolation |
| 4 | Agents & Tools | `@tool`, `create_react_agent` (LangGraph), custom tools |
| 5 | RAG | Embeddings, `InMemoryVectorStore`, RAG chain, source citations, conversational RAG |

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/Apusapati-9/langchain-learning.git
cd langchain-learning
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your OpenAI API key**
```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

## Usage

```bash
# Interactive menu
python main.py

# Run a specific lesson
python main.py --lesson 1

# Run all lessons
python main.py --all
```

## Project Structure

```
langchain-learning/
├── main.py                  # CLI entry point
├── requirements.txt
├── .env.example
└── lessons/
    ├── 01_llm_prompts.py    # Lesson 1: LLMs & Prompt Templates
    ├── 02_chains.py         # Lesson 2: Chains (LCEL)
    ├── 03_memory.py         # Lesson 3: Memory & Chat History
    ├── 04_agents.py         # Lesson 4: Agents & Tools
    └── 05_rag.py            # Lesson 5: RAG
```

## Requirements

- Python 3.10+
- OpenAI API key
