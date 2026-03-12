"""
Lesson 3: Memory & Chat History
---------------------------------
Covers:
  - In-memory ChatMessageHistory
  - RunnableWithMessageHistory  (modern LCEL approach)
  - Multi-session memory (different session_ids)
  - Manually inspecting stored messages
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


# Simple in-memory store keyed by session_id
_store: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]


def build_chain(llm: ChatOpenAI) -> RunnableWithMessageHistory:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer concisely."),
        MessagesPlaceholder(variable_name="history"),   # injected by RunnableWithMessageHistory
        ("human", "{input}"),
    ])
    chain = prompt | llm
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )


def chat(chain: RunnableWithMessageHistory, session_id: str, user_input: str) -> str:
    response = chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )
    return response.content


def demo_single_session(chain: RunnableWithMessageHistory) -> None:
    print("\n--- 3a. Single session conversation ---")
    session = "session-alice"
    turns = [
        "My name is Alice and I love Python.",
        "What's my name and favourite language?",
        "Give me a Python tip related to what I love.",
    ]
    for user_msg in turns:
        print(f"\nUser : {user_msg}")
        reply = chat(chain, session, user_msg)
        print(f"Bot  : {reply}")


def demo_multi_session(chain: RunnableWithMessageHistory) -> None:
    print("\n--- 3b. Two isolated sessions ---")
    chat(chain, "session-bob",   "My favourite language is JavaScript.")
    chat(chain, "session-carol", "My favourite language is Rust.")

    print("Bob's session   →", chat(chain, "session-bob",   "What language do I like?"))
    print("Carol's session →", chat(chain, "session-carol", "What language do I like?"))


def demo_inspect_history() -> None:
    print("\n--- 3c. Inspect stored messages ---")
    for session_id, history in _store.items():
        print(f"\n[{session_id}]")
        for msg in history.messages:
            role = msg.__class__.__name__.replace("Message", "")
            print(f"  {role}: {msg.content[:80]}{'...' if len(msg.content) > 80 else ''}")


def run():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    chain = build_chain(llm)
    demo_single_session(chain)
    demo_multi_session(chain)
    demo_inspect_history()
