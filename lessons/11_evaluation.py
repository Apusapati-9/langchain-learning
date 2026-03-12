"""
Lesson 11: Evaluation & Testing
---------------------------------
Covers:
  - LLM-as-judge (custom scoring chain)
  - Criteria-based evaluation (correctness, relevance, conciseness)
  - Pairwise comparison (A vs B)
  - Dataset-level batch evaluation
  - Building a simple eval harness
"""

from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


# ===========================================================================
# Shared Pydantic models
# ===========================================================================

class ScoreResult(BaseModel):
    score: int        = Field(description="Score from 1 (poor) to 5 (excellent)")
    reasoning: str    = Field(description="One-sentence explanation of the score")


class CriteriaResult(BaseModel):
    correctness:  int = Field(description="Correctness score 1-5")
    relevance:    int = Field(description="Relevance score 1-5")
    conciseness:  int = Field(description="Conciseness score 1-5")
    overall:      int = Field(description="Overall score 1-5")
    feedback:     str = Field(description="One sentence of overall feedback")


class PairwiseResult(BaseModel):
    winner: str       = Field(description="'A', 'B', or 'tie'")
    reasoning: str    = Field(description="One-sentence explanation of the choice")


# ===========================================================================
# 11a. LLM-as-judge — single response scoring
# ===========================================================================

def demo_llm_judge(llm: ChatOpenAI) -> None:
    print("\n--- 11a. LLM-as-judge (score 1–5) ---")
    parser = PydanticOutputParser(pydantic_object=ScoreResult)

    prompt = ChatPromptTemplate.from_template(
        "You are an impartial evaluator. Score the response to the given question.\n\n"
        "Question : {question}\n"
        "Response : {response}\n\n"
        "{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser

    examples = [
        {
            "question": "What is a Python decorator?",
            "response": "A decorator is a function that wraps another function to add behaviour.",
        },
        {
            "question": "What is a Python decorator?",
            "response": "It makes Python better.",
        },
    ]
    for ex in examples:
        result: ScoreResult = chain.invoke(ex)
        print(f"\n  Response  : {ex['response'][:60]}")
        print(f"  Score     : {result.score}/5")
        print(f"  Reasoning : {result.reasoning}")


# ===========================================================================
# 11b. Criteria-based evaluation
# ===========================================================================

def demo_criteria_eval(llm: ChatOpenAI) -> None:
    print("\n--- 11b. Criteria-based evaluation ---")
    parser = PydanticOutputParser(pydantic_object=CriteriaResult)

    prompt = ChatPromptTemplate.from_template(
        "Evaluate the response on four criteria, each scored 1–5.\n\n"
        "Question : {question}\n"
        "Reference: {reference}\n"
        "Response : {response}\n\n"
        "{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser

    result: CriteriaResult = chain.invoke({
        "question":  "Explain what RAG is.",
        "reference": "RAG retrieves relevant documents and uses them as context for an LLM.",
        "response":  (
            "RAG stands for Retrieval-Augmented Generation. It first retrieves relevant "
            "documents from a knowledge base, then passes them as context to the LLM to "
            "generate a grounded answer, reducing hallucinations."
        ),
    })
    print(f"  Correctness  : {result.correctness}/5")
    print(f"  Relevance    : {result.relevance}/5")
    print(f"  Conciseness  : {result.conciseness}/5")
    print(f"  Overall      : {result.overall}/5")
    print(f"  Feedback     : {result.feedback}")


# ===========================================================================
# 11c. Pairwise comparison (A vs B)
# ===========================================================================

def demo_pairwise(llm: ChatOpenAI) -> None:
    print("\n--- 11c. Pairwise comparison (A vs B) ---")
    parser = PydanticOutputParser(pydantic_object=PairwiseResult)

    prompt = ChatPromptTemplate.from_template(
        "Compare two responses to the question. Pick the better one or call a tie.\n\n"
        "Question   : {question}\n"
        "Response A : {response_a}\n"
        "Response B : {response_b}\n\n"
        "{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser

    result: PairwiseResult = chain.invoke({
        "question":   "What is the GIL in Python?",
        "response_a": "The GIL is a mutex that prevents multiple native threads from executing Python bytecodes at once.",
        "response_b": "GIL stands for Global Interpreter Lock. It's a thing in Python.",
    })
    print(f"  Winner    : Response {result.winner}")
    print(f"  Reasoning : {result.reasoning}")


# ===========================================================================
# 11d. Batch dataset evaluation
# ===========================================================================

@dataclass
class EvalSample:
    question:  str
    reference: str
    response:  str


def demo_batch_eval(llm: ChatOpenAI) -> None:
    print("\n--- 11d. Batch dataset evaluation ---")
    parser = PydanticOutputParser(pydantic_object=ScoreResult)

    prompt = ChatPromptTemplate.from_template(
        "Score this response 1–5 for correctness vs the reference.\n\n"
        "Question : {question}\n"
        "Reference: {reference}\n"
        "Response : {response}\n\n"
        "{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser

    dataset = [
        EvalSample(
            question="What year was Python created?",
            reference="Python was first released in 1991.",
            response="Python was created in 1991 by Guido van Rossum.",
        ),
        EvalSample(
            question="What is LCEL?",
            reference="LCEL is LangChain Expression Language using the pipe operator.",
            response="LCEL is a way to make coffee with LangChain.",
        ),
        EvalSample(
            question="What does RAG stand for?",
            reference="Retrieval-Augmented Generation.",
            response="Retrieval-Augmented Generation — it retrieves docs and uses them as LLM context.",
        ),
    ]

    scores = []
    print(f"\n  {'Question':<40} {'Score':>5}  Reasoning")
    print(f"  {'-'*40} {'-'*5}  {'-'*30}")
    for sample in dataset:
        result: ScoreResult = chain.invoke({
            "question":  sample.question,
            "reference": sample.reference,
            "response":  sample.response,
        })
        scores.append(result.score)
        print(f"  {sample.question:<40} {result.score:>5}/5  {result.reasoning[:50]}")

    avg = sum(scores) / len(scores)
    print(f"\n  Average score: {avg:.1f}/5")


# ===========================================================================
# Entry point
# ===========================================================================

def run():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    demo_llm_judge(llm)
    demo_criteria_eval(llm)
    demo_pairwise(llm)
    demo_batch_eval(llm)
