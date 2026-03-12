"""
Lesson 1: LLMs & Prompt Templates
----------------------------------
Covers:
  - Direct LLM / ChatModel invocation
  - PromptTemplate  (string → string)
  - ChatPromptTemplate (messages → messages)
  - Few-shot prompting
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage


def demo_direct_invocation(llm: ChatOpenAI) -> None:
    print("\n--- 1a. Direct invocation with message list ---")
    messages = [
        SystemMessage(content="You are a helpful Python tutor."),
        HumanMessage(content="What is a list comprehension? Give a one-line example."),
    ]
    response = llm.invoke(messages)
    print("Response:", response.content)


def demo_prompt_template(llm: ChatOpenAI) -> None:
    print("\n--- 1b. PromptTemplate (string template) ---")
    template = PromptTemplate(
        input_variables=["topic", "level"],
        template="Explain {topic} to a {level} programmer in 2 sentences.",
    )
    prompt = template.invoke({"topic": "decorators", "level": "beginner"})
    print("Formatted prompt:\n", prompt.text)

    # Pipe the template directly into the LLM using LCEL
    chain = template | llm
    result = chain.invoke({"topic": "decorators", "level": "beginner"})
    print("LLM response:", result.content)


def demo_chat_prompt_template(llm: ChatOpenAI) -> None:
    print("\n--- 1c. ChatPromptTemplate ---")
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in {language} programming."),
        ("human", "Give me a quick tip about {topic}."),
    ])
    chain = chat_prompt | llm
    result = chain.invoke({"language": "Python", "topic": "error handling"})
    print("Response:", result.content)


def demo_few_shot(llm: ChatOpenAI) -> None:
    print("\n--- 1d. Few-Shot Prompting ---")
    examples = [
        {"input": "happy", "output": "sad"},
        {"input": "tall",  "output": "short"},
        {"input": "fast",  "output": "slow"},
    ]
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="Word: {input} → Antonym: {output}",
    )
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="Give the antonym of each word.",
        suffix="Word: {input} → Antonym:",
        input_variables=["input"],
    )
    chain = few_shot_prompt | llm
    result = chain.invoke({"input": "bright"})
    print("Antonym of 'bright':", result.content.strip())


def run():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    demo_direct_invocation(llm)
    demo_prompt_template(llm)
    demo_chat_prompt_template(llm)
    demo_few_shot(llm)
