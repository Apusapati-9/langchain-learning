"""
Lesson 2: Chains with LCEL (LangChain Expression Language)
------------------------------------------------------------
Covers:
  - The pipe operator  |  to compose runnables
  - StrOutputParser
  - Branching with RunnableBranch
  - Parallel execution with RunnableParallel
  - Passing through data with RunnablePassthrough
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableBranch


def demo_basic_chain(llm: ChatOpenAI) -> None:
    print("\n--- 2a. Basic LCEL chain: prompt | llm | parser ---")
    prompt = ChatPromptTemplate.from_template(
        "Summarize this text in one sentence:\n\n{text}"
    )
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "text": (
            "LangChain Expression Language (LCEL) is a declarative way to compose "
            "chains. It uses the pipe operator to connect runnables, making it easy "
            "to build complex pipelines from simple components."
        )
    })
    print("Summary:", result)


def demo_sequential_chain(llm: ChatOpenAI) -> None:
    print("\n--- 2b. Sequential chain (translate → formal) ---")
    parser = StrOutputParser()

    translate_prompt = ChatPromptTemplate.from_template(
        "Translate the following to English:\n{text}"
    )
    formalize_prompt = ChatPromptTemplate.from_template(
        "Rewrite this in a formal tone:\n{text}"
    )

    # Step 1: translate
    translate_chain = translate_prompt | llm | parser
    # Step 2: pass translated text into formalize
    full_chain = translate_chain | (lambda translated: {"text": translated}) | formalize_prompt | llm | parser

    result = full_chain.invoke({"text": "hola, ¿cómo estás? todo bien por acá, trabajando mucho."})
    print("Formalized English:", result)


def demo_parallel(llm: ChatOpenAI) -> None:
    print("\n--- 2c. RunnableParallel (run two chains at once) ---")
    parser = StrOutputParser()
    topic = {"topic": RunnablePassthrough()}

    pros_chain = (
        ChatPromptTemplate.from_template("List 2 pros of {topic}.") | llm | parser
    )
    cons_chain = (
        ChatPromptTemplate.from_template("List 2 cons of {topic}.") | llm | parser
    )

    parallel = RunnableParallel(pros=pros_chain, cons=cons_chain)
    result = parallel.invoke("remote work")
    print("Pros:\n", result["pros"])
    print("Cons:\n", result["cons"])


def demo_passthrough(llm: ChatOpenAI) -> None:
    print("\n--- 2d. RunnablePassthrough (augment context) ---")
    parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_template(
        "Given this context: {context}\n\nAnswer: {question}"
    )
    chain = (
        RunnablePassthrough()  # passes the input dict through unchanged
        | prompt
        | llm
        | parser
    )
    result = chain.invoke({
        "context": "Python 3.12 introduced the new 'type' statement for type aliases.",
        "question": "What did Python 3.12 add?",
    })
    print("Answer:", result)


def run():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    demo_basic_chain(llm)
    demo_sequential_chain(llm)
    demo_parallel(llm)
    demo_passthrough(llm)
