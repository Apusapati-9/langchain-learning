"""
LangChain Learning App
======================
An interactive CLI to explore core LangChain concepts.

Usage:
  python main.py           # interactive menu
  python main.py --all     # run all lessons sequentially
  python main.py --lesson 2  # run a specific lesson (1-4)
"""

import argparse
import sys
from dotenv import load_dotenv

load_dotenv()  # load OPENAI_API_KEY from .env

LESSONS = {
    1: ("LLMs & Prompt Templates",  "lessons.01_llm_prompts"),
    2: ("Chains (LCEL)",             "lessons.02_chains"),
    3: ("Memory & Chat History",     "lessons.03_memory"),
    4: ("Agents & Tools",            "lessons.04_agents"),
    5: ("RAG",                       "lessons.05_rag"),
    6: ("LangGraph",                 "lessons.06_langgraph"),
    7: ("Output Parsers",            "lessons.07_output_parsers"),
    8: ("Streaming",                 "lessons.08_streaming"),
    9: ("Document Loaders",          "lessons.09_document_loaders"),
   10: ("Multi-Agent Systems",       "lessons.10_multi_agent"),
   11: ("Evaluation & Testing",      "lessons.11_evaluation"),
   12: ("Caching",                   "lessons.12_caching"),
   13: ("Callbacks & Tracing",       "lessons.13_callbacks"),
   14: ("LangChain + FastAPI",       "lessons.14_fastapi"),
   15: ("LangChain + Databases",     "lessons.15_databases"),
}

BANNER = """
╔══════════════════════════════════════════════╗
║        LangChain Learning App  🦜🔗          ║
╚══════════════════════════════════════════════╝
"""


def run_lesson(number: int) -> None:
    name, module_path = LESSONS[number]
    print(f"\n{'#'*56}")
    print(f"  Lesson {number}: {name}")
    print(f"{'#'*56}")
    import importlib
    module = importlib.import_module(module_path)
    module.run()
    print(f"\n[Lesson {number} complete]")


def interactive_menu() -> None:
    print(BANNER)
    while True:
        print("\nAvailable lessons:")
        for num, (name, _) in LESSONS.items():
            print(f"  [{num}] {name}")
        print("  [a] Run all lessons")
        print("  [q] Quit")

        choice = input("\nEnter choice: ").strip().lower()

        if choice == "q":
            print("Goodbye!")
            sys.exit(0)
        elif choice == "a":
            for num in LESSONS:
                run_lesson(num)
        elif choice.isdigit() and int(choice) in LESSONS:
            run_lesson(int(choice))
        else:
            print("Invalid choice, please try again.")


def main() -> None:
    parser = argparse.ArgumentParser(description="LangChain Learning App")
    parser.add_argument("--all", action="store_true", help="Run all lessons")
    parser.add_argument("--lesson", type=int, choices=list(LESSONS.keys()),
                        help="Run a specific lesson (1-4)")
    args = parser.parse_args()

    if args.all:
        print(BANNER)
        for num in LESSONS:
            run_lesson(num)
    elif args.lesson:
        print(BANNER)
        run_lesson(args.lesson)
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
