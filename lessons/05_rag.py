"""
Lesson 5: RAG (Retrieval-Augmented Generation)
------------------------------------------------
Covers:
  - Creating and splitting Documents
  - Embedding text with OpenAIEmbeddings
  - Storing & querying with InMemoryVectorStore
  - Basic RAG chain (retrieve → prompt → generate)
  - RAG with source citations
  - Conversational RAG (memory + retrieval)
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# Sample knowledge base (stand-in for real documents / PDFs / web pages)
# ---------------------------------------------------------------------------

DOCUMENTS = [
    Document(
        page_content=(
            "LangChain is a framework for building applications powered by large language models. "
            "It provides abstractions for chaining LLM calls, managing prompts, and integrating "
            "external tools and data sources. LangChain supports Python and JavaScript."
        ),
        metadata={"source": "langchain_overview.txt", "topic": "LangChain"},
    ),
    Document(
        page_content=(
            "LangChain Expression Language (LCEL) is the recommended way to build chains in "
            "LangChain. It uses the pipe operator | to connect runnables. LCEL supports streaming, "
            "async execution, batching, and parallel execution out of the box."
        ),
        metadata={"source": "lcel_guide.txt", "topic": "LCEL"},
    ),
    Document(
        page_content=(
            "RAG (Retrieval-Augmented Generation) is a technique that enhances LLM responses by "
            "first retrieving relevant documents from a knowledge base, then passing those documents "
            "as context to the LLM. This reduces hallucinations and keeps answers grounded in facts."
        ),
        metadata={"source": "rag_concepts.txt", "topic": "RAG"},
    ),
    Document(
        page_content=(
            "Vector stores are databases optimised for similarity search. Text is converted into "
            "numerical vectors (embeddings) using an embedding model. At query time, the query is "
            "also embedded and the most similar document vectors are retrieved. Popular vector stores "
            "include FAISS, Chroma, Pinecone, and Weaviate."
        ),
        metadata={"source": "vector_stores.txt", "topic": "Vector Stores"},
    ),
    Document(
        page_content=(
            "LangGraph is a library built on top of LangChain for building stateful, multi-actor "
            "applications with LLMs. It models workflows as graphs where nodes are functions and "
            "edges define control flow. LangGraph replaced the older AgentExecutor in LangChain 1.x."
        ),
        metadata={"source": "langgraph_intro.txt", "topic": "LangGraph"},
    ),
    Document(
        page_content=(
            "Embeddings are dense numerical representations of text. OpenAI's text-embedding-3-small "
            "model produces 1536-dimensional vectors. Similar pieces of text have vectors that are "
            "close together in vector space, enabling semantic (meaning-based) search rather than "
            "simple keyword matching."
        ),
        metadata={"source": "embeddings_explained.txt", "topic": "Embeddings"},
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_vectorstore(embeddings: OpenAIEmbeddings) -> InMemoryVectorStore:
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40)
    chunks = splitter.split_documents(DOCUMENTS)
    print(f"  [Vector store] {len(DOCUMENTS)} documents → {len(chunks)} chunks after splitting")
    vectorstore = InMemoryVectorStore(embeddings)
    vectorstore.add_documents(chunks)
    return vectorstore


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo_similarity_search(vectorstore: InMemoryVectorStore) -> None:
    print("\n--- 5a. Similarity search (retriever) ---")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    query = "How does semantic search work with embeddings?"
    results = retriever.invoke(query)
    print(f"Query : {query}")
    for i, doc in enumerate(results, 1):
        print(f"\n  Result {i} [{doc.metadata['source']}]:")
        print(f"  {doc.page_content[:120]}...")


def demo_basic_rag(llm: ChatOpenAI, vectorstore: InMemoryVectorStore) -> None:
    print("\n--- 5b. Basic RAG chain ---")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        "Use only the context below to answer the question. "
        "If the answer is not in the context, say 'I don't know'.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    questions = [
        "What is LCEL and what does it support?",
        "What is LangGraph and why was it introduced?",
        "What is the capital of France?",   # out-of-context question
    ]
    for q in questions:
        print(f"\n  Q: {q}")
        print(f"  A: {rag_chain.invoke(q)}")


def demo_rag_with_sources(llm: ChatOpenAI, vectorstore: InMemoryVectorStore) -> None:
    print("\n--- 5c. RAG with source citations ---")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    prompt = ChatPromptTemplate.from_template(
        "Answer the question using the provided context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )

    # Keep docs alongside the answer by using RunnablePassthrough.assign
    from langchain_core.runnables import RunnablePassthrough

    def retrieve_with_sources(question: str) -> dict:
        docs = retriever.invoke(question)
        context = format_docs(docs)
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        sources = list({doc.metadata["source"] for doc in docs})
        return {"answer": answer, "sources": sources}

    query = "What are vector stores and give examples?"
    result = retrieve_with_sources(query)
    print(f"\n  Q: {query}")
    print(f"  A: {result['answer']}")
    print(f"  Sources: {result['sources']}")


def demo_conversational_rag(llm: ChatOpenAI, vectorstore: InMemoryVectorStore) -> None:
    print("\n--- 5d. Conversational RAG (memory + retrieval) ---")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    store: dict[str, ChatMessageHistory] = {}

    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Use the retrieved context to answer questions.\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    def retrieve_context(inputs: dict) -> dict:
        docs = retriever.invoke(inputs["question"])
        return {**inputs, "context": format_docs(docs)}

    chain = retrieve_context | prompt | llm | StrOutputParser()

    conv_chain = RunnableWithMessageHistory(
        chain,                                          # type: ignore[arg-type]
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    def chat(question: str) -> str:
        return conv_chain.invoke(
            {"question": question},
            config={"configurable": {"session_id": "rag-session"}},
        )

    turns = [
        "What is RAG and why is it useful?",
        "How do embeddings relate to what you just explained?",
        "Summarise both topics in one sentence.",
    ]
    for turn in turns:
        print(f"\n  User : {turn}")
        print(f"  Bot  : {chat(turn)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    print("\nBuilding vector store from knowledge base...")
    vectorstore = build_vectorstore(embeddings)

    demo_similarity_search(vectorstore)
    demo_basic_rag(llm, vectorstore)
    demo_rag_with_sources(llm, vectorstore)
    demo_conversational_rag(llm, vectorstore)
