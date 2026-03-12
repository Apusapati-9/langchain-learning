"""
Lesson 7: Output Parsers
-------------------------
Covers:
  - StrOutputParser       — plain text
  - CommaSeparatedListOutputParser — simple list
  - JsonOutputParser      — free-form JSON
  - PydanticOutputParser  — structured, validated output
  - with_structured_output — native structured output (OpenAI function calling)
  - OutputFixingParser    — auto-repair malformed output
"""

from typing import Optional
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    CommaSeparatedListOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
)
from langchain_core.messages import HumanMessage


# ===========================================================================
# 7a. StrOutputParser — raw string from AIMessage
# ===========================================================================

def demo_str_parser(llm: ChatOpenAI) -> None:
    print("\n--- 7a. StrOutputParser ---")
    chain = (
        ChatPromptTemplate.from_template("Name the creator of Python in one sentence.")
        | llm
        | StrOutputParser()
    )
    result = chain.invoke({})
    print("Type   :", type(result).__name__)
    print("Output :", result)


# ===========================================================================
# 7b. CommaSeparatedListOutputParser
# ===========================================================================

def demo_list_parser(llm: ChatOpenAI) -> None:
    print("\n--- 7b. CommaSeparatedListOutputParser ---")
    parser = CommaSeparatedListOutputParser()

    prompt = PromptTemplate(
        template="List 5 popular Python libraries for {domain}.\n{format_instructions}",
        input_variables=["domain"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser
    result = chain.invoke({"domain": "data science"})
    print("Type   :", type(result).__name__)
    print("Items  :")
    for i, item in enumerate(result, 1):
        print(f"  {i}. {item.strip()}")


# ===========================================================================
# 7c. JsonOutputParser — free-form JSON dict
# ===========================================================================

def demo_json_parser(llm: ChatOpenAI) -> None:
    print("\n--- 7c. JsonOutputParser ---")
    parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_template(
        "Return a JSON object with keys 'language', 'year_created', and 'use_case' "
        "for the programming language: {language}\n{format_instructions}"
    )
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser
    result = chain.invoke({"language": "Rust"})
    print("Type   :", type(result).__name__)
    for k, v in result.items():
        print(f"  {k}: {v}")


# ===========================================================================
# 7d. PydanticOutputParser — validated structured output
# ===========================================================================

class MovieReview(BaseModel):
    title:       str   = Field(description="Title of the movie")
    genre:       str   = Field(description="Genre of the movie")
    rating:      float = Field(description="Rating out of 10")
    summary:     str   = Field(description="One-sentence summary")
    recommended: bool  = Field(description="Whether you recommend it")


def demo_pydantic_parser(llm: ChatOpenAI) -> None:
    print("\n--- 7d. PydanticOutputParser ---")
    parser = PydanticOutputParser(pydantic_object=MovieReview)

    prompt = PromptTemplate(
        template=(
            "Write a short review for the movie '{movie}'.\n"
            "{format_instructions}"
        ),
        input_variables=["movie"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser
    result: MovieReview = chain.invoke({"movie": "Inception"})

    print("Type      :", type(result).__name__)
    print(f"Title     : {result.title}")
    print(f"Genre     : {result.genre}")
    print(f"Rating    : {result.rating}/10")
    print(f"Summary   : {result.summary}")
    print(f"Recommend : {result.recommended}")


# ===========================================================================
# 7e. with_structured_output — native OpenAI function calling
# ===========================================================================

class WeatherReport(BaseModel):
    city:           str   = Field(description="Name of the city")
    temperature_c:  float = Field(description="Temperature in Celsius")
    condition:      str   = Field(description="Weather condition, e.g. sunny, rainy")
    humidity_pct:   int   = Field(description="Humidity percentage")
    advice:         str   = Field(description="One-sentence advice for the day")


def demo_structured_output(llm: ChatOpenAI) -> None:
    print("\n--- 7e. with_structured_output (native function calling) ---")
    structured_llm = llm.with_structured_output(WeatherReport)

    prompt = ChatPromptTemplate.from_template(
        "Invent a realistic-sounding weather report for {city} today."
    )
    chain = prompt | structured_llm
    result: WeatherReport = chain.invoke({"city": "Tokyo"})

    print("Type          :", type(result).__name__)
    print(f"City          : {result.city}")
    print(f"Temperature   : {result.temperature_c}°C")
    print(f"Condition     : {result.condition}")
    print(f"Humidity      : {result.humidity_pct}%")
    print(f"Advice        : {result.advice}")


# ===========================================================================
# 7f. Manual output fixing — repair malformed output with a follow-up LLM call
# ===========================================================================

class BookInfo(BaseModel):
    title:  str = Field(description="Book title")
    author: str = Field(description="Author name")
    year:   int = Field(description="Publication year")


def demo_fixing_parser(llm: ChatOpenAI) -> None:
    print("\n--- 7f. Manual output fixing (repair malformed JSON) ---")
    base_parser = PydanticOutputParser(pydantic_object=BookInfo)

    # Intentionally malformed JSON — unquoted keys, wrong field name
    bad_output = '{title: "1984", author: George Orwell, publication_year: 1949}'
    print("Bad output  :", bad_output)

    try:
        base_parser.parse(bad_output)
    except Exception as e:
        print(f"Parse error : {type(e).__name__} — attempting fix...")

        fix_prompt = (
            f"The following output is malformed. Fix it to match this schema:\n"
            f"{base_parser.get_format_instructions()}\n\n"
            f"Malformed output:\n{bad_output}\n\n"
            f"Return only the corrected JSON, nothing else."
        )
        fixed_raw = llm.invoke([HumanMessage(content=fix_prompt)]).content
        fixed: BookInfo = base_parser.parse(fixed_raw)
        print(f"Fixed result: title='{fixed.title}', author='{fixed.author}', year={fixed.year}")


# ===========================================================================
# Entry point
# ===========================================================================

def run():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    demo_str_parser(llm)
    demo_list_parser(llm)
    demo_json_parser(llm)
    demo_pydantic_parser(llm)
    demo_structured_output(llm)
    demo_fixing_parser(llm)
