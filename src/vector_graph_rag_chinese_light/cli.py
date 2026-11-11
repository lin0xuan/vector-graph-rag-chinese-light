"""Command-line ingestion and querying."""
import argparse
import json
from urllib.error import HTTPError

from .rag import ChineseLightRAG


def main():
    parser = argparse.ArgumentParser(description="Lightweight Chinese vector graph RAG")
    parser.add_argument("--data-dir", default=".vgrag", help="Directory for the local index manifest")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Index or replace a document")
    ingest.add_argument("path")
    ingest.add_argument("--source", help="Stable identifier used to replace or remove the document")
    ask = commands.add_parser("ask", help="Answer a question and return source excerpts")
    ask.add_argument("question")
    remove = commands.add_parser("remove", help="Remove a source from the searchable index")
    remove.add_argument("source")

    args = parser.parse_args()
    try:
        rag = ChineseLightRAG(args.data_dir)
        if args.command == "ingest":
            result = {"passages": rag.upsert_file(args.path, source=args.source)}
        elif args.command == "ask":
            result = rag.query(args.question).as_dict()
        else:
            result = {"removed": rag.remove_source(args.source)}
    except HTTPError as error:
        parser.exit(1, f"Upstream HTTP {error.code}. Check the Chroma URL and model configuration.\n")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
