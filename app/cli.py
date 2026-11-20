"""Command-line entry point: `python -m app.cli --help`."""
import argparse
import sys
from typing import Optional

from app.core import get_agricultural_advice


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="crop-advice-snippet-ai",
        description="Farmers use a CLI tool to generate concise, context-aware agricultural advice snippets for specific crops, trained on a dataset of farming best practices.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")

    # Milestone 6 command
    advice_parser = sub.add_parser("advice", help="Get agricultural advice snippet for a crop or problem")
    advice_parser.add_argument("crop", nargs="?", default="wheat", help="Crop name (e.g., wheat, rice, tomato, corn, potato)")
    advice_parser.add_argument("--problem", "-p", default=None, help="Specific problem, pest, or disease (e.g. rust, blight)")

    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Crop Advice Snippet AI is ready. Open README.md for your milestones.")
    elif args.command == "advice":
        advice = get_agricultural_advice(args.crop, args.problem)
        print(advice)


if __name__ == "__main__":
    main()
