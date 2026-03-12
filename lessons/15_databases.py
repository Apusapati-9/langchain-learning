"""
Lesson 15: LangChain + Databases
----------------------------------
Covers:
  - Creating a SQLite database with sample data
  - SQLDatabase utility — inspect schema
  - Natural language → SQL with a prompt chain
  - SQL agent (LLM decides which queries to run)
  - Read-only safety guard
"""

import sqlite3
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage


DB_PATH = Path(__file__).parent.parent / "data" / "store.db"


# ---------------------------------------------------------------------------
# Setup — create and seed the SQLite database
# ---------------------------------------------------------------------------

def create_database() -> None:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.executescript("""
        DROP TABLE IF EXISTS products;
        DROP TABLE IF EXISTS customers;
        DROP TABLE IF EXISTS orders;

        CREATE TABLE products (
            id       INTEGER PRIMARY KEY,
            name     TEXT NOT NULL,
            category TEXT NOT NULL,
            price    REAL NOT NULL,
            stock    INTEGER NOT NULL
        );

        CREATE TABLE customers (
            id      INTEGER PRIMARY KEY,
            name    TEXT NOT NULL,
            email   TEXT NOT NULL,
            country TEXT NOT NULL
        );

        CREATE TABLE orders (
            id          INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(id),
            product_id  INTEGER REFERENCES products(id),
            quantity    INTEGER NOT NULL,
            order_date  TEXT NOT NULL
        );
    """)

    cur.executemany("INSERT INTO products VALUES (?,?,?,?,?)", [
        (1, "Python Book",       "Books",       29.99, 150),
        (2, "Mechanical Keyboard","Electronics", 89.99,  45),
        (3, "Desk Lamp",         "Electronics",  19.99,  80),
        (4, "Notebook",          "Stationery",    4.99, 300),
        (5, "Coffee Mug",        "Kitchen",       9.99, 200),
        (6, "JavaScript Book",   "Books",        24.99,  90),
    ])

    cur.executemany("INSERT INTO customers VALUES (?,?,?,?)", [
        (1, "Alice",   "alice@example.com",   "USA"),
        (2, "Bob",     "bob@example.com",     "UK"),
        (3, "Carol",   "carol@example.com",   "Germany"),
        (4, "Dave",    "dave@example.com",    "USA"),
        (5, "Eve",     "eve@example.com",     "France"),
    ])

    cur.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", [
        (1, 1, 1, 2, "2025-01-10"),
        (2, 1, 3, 1, "2025-01-15"),
        (3, 2, 2, 1, "2025-01-20"),
        (4, 3, 4, 5, "2025-02-01"),
        (5, 4, 1, 1, "2025-02-05"),
        (6, 5, 5, 3, "2025-02-10"),
        (7, 2, 6, 1, "2025-02-12"),
        (8, 1, 5, 2, "2025-03-01"),
    ])

    conn.commit()
    conn.close()
    print(f"  Database created : {DB_PATH}")


# ===========================================================================
# 15a. Inspect schema with SQLDatabase
# ===========================================================================

def demo_schema_inspection(db: SQLDatabase) -> None:
    print("\n--- 15a. Schema inspection ---")
    print(f"  Tables    : {db.get_usable_table_names()}")
    print(f"  Table info:\n{db.get_table_info()[:600]}...")


# ===========================================================================
# 15b. Natural language → SQL (manual chain)
# ===========================================================================

def demo_nl_to_sql(llm: ChatOpenAI, db: SQLDatabase) -> None:
    print("\n--- 15b. Natural language → SQL ---")

    nl_to_sql_prompt = ChatPromptTemplate.from_template(
        "Given this SQLite schema:\n{schema}\n\n"
        "Write a single SQL SELECT query (no explanation) to answer:\n{question}"
    )
    sql_chain = nl_to_sql_prompt | llm | StrOutputParser()

    questions = [
        "How many products are in the Electronics category?",
        "Which customer has placed the most orders?",
        "What is the total revenue from book sales?",
    ]

    schema = db.get_table_info()
    for question in questions:
        sql = sql_chain.invoke({"schema": schema, "question": question}).strip()
        # Strip markdown fences if present
        if sql.startswith("```"):
            sql = "\n".join(sql.split("\n")[1:-1])
        try:
            result = db.run(sql)
        except Exception as e:
            result = f"Error: {e}"
        print(f"\n  Q  : {question}")
        print(f"  SQL: {sql}")
        print(f"  →  : {result}")


# ===========================================================================
# 15c. SQL agent — LLM decides which queries to run
# ===========================================================================

def demo_sql_agent(llm: ChatOpenAI, db: SQLDatabase) -> None:
    print("\n--- 15c. SQL agent (LLM-driven query loop) ---")

    @tool
    def run_sql_query(query: str) -> str:
        """
        Execute a read-only SQL SELECT query against the store database.
        Only SELECT statements are allowed.
        """
        query = query.strip()
        if not query.upper().startswith("SELECT"):
            return "Error: Only SELECT queries are allowed."
        try:
            return db.run(query)
        except Exception as e:
            return f"SQL Error: {e}"

    @tool
    def get_schema() -> str:
        """Return the database schema (table names and columns)."""
        return db.get_table_info()

    agent = create_react_agent(
        llm,
        tools=[run_sql_query, get_schema],
        prompt=(
            "You are a data analyst with access to a SQLite store database. "
            "Use get_schema to understand the tables, then run_sql_query to answer questions. "
            "Only use SELECT statements. Always provide a clear final answer."
        ),
    )

    queries = [
        "What are the top 3 most expensive products?",
        "Which country has the most customers?",
    ]
    for query in queries:
        print(f"\n  Query  : {query}")
        result = agent.invoke({"messages": [HumanMessage(content=query)]})
        print(f"  Answer : {result['messages'][-1].content}")


# ===========================================================================
# Entry point
# ===========================================================================

def run():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    print("\nSetting up database...")
    create_database()

    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

    demo_schema_inspection(db)
    demo_nl_to_sql(llm, db)
    demo_sql_agent(llm, db)
