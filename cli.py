#!/usr/bin/env python3
"""
Search Framework CLI

Usage examples:

  # Ingest all sources defined in the config
  python cli.py ingest configs/hs-aalen.yaml

  # Ingest only specific sources (by name)
  python cli.py ingest configs/hs-aalen.yaml --source hs_aalen_main --source asta

  # Start the search API server
  python cli.py serve configs/hs-aalen.yaml

  # Start on a custom port
  python cli.py serve configs/hs-aalen.yaml --port 8080 --host 0.0.0.0

  # Create a new config from template
  python cli.py new meine-firma

  # Run evaluation against the API
  python cli.py eval configs/hs-aalen.yaml
"""

import sys
import logging
import argparse
from pathlib import Path

# Setup logging for the CLI
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cli")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_ingest(args):
    """Run the ingestion pipeline for a config."""
    from framework.ingestion.pipeline import SearchPipeline

    config_path = Path(args.config)
    if not config_path.exists():
        log.error("Config not found: %s", config_path)
        sys.exit(1)

    pipeline = SearchPipeline(config_path)

    source_filter = list(args.source) if args.source else None
    if source_filter:
        log.info("Filtering to sources: %s", source_filter)

    pipeline.run(source_names=source_filter)


def cmd_serve(args):
    """Start the FastAPI search server."""
    try:
        import uvicorn
    except ImportError:
        log.error("uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        log.error("Config not found: %s", config_path)
        sys.exit(1)

    from framework.app import create_app
    app = create_app(str(config_path))

    log.info("Starting server for config: %s", config_path)
    log.info("Listening on http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


def cmd_new(args):
    """Create a new config file from the template."""
    name = args.name
    dest = Path(f"configs/{name}.yaml")
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not args.force:
        log.error("Config already exists: %s  (use --force to overwrite)", dest)
        sys.exit(1)

    template = f"""# ============================================================
# {name} — Search Engine Configuration
# ============================================================

name: "{name}"
language: "de"
description: "Search engine for {name}"

embedding_model: "intfloat/multilingual-e5-base"
vector_size: 768

sparse_vectors: false
sparse_model: "Qdrant/bm42-all-minilm-l6-v2-attentions"

qdrant:
  host: "localhost"
  port: 6333

llm:
  ollama_url: "http://localhost:11434/api/generate"
  ollama_model: "deepseek-r1:8b"
  openai_model: "gpt-4o-mini"

sources:
  # Website source — uncomment and fill in your URL
  # - name: "main"
  #   type: website
  #   url: "https://example.com"
  #   max_depth: 3
  #   collection: "{name}_main"
  #   result_type: "webpage"
  #   search_weight: 1.0

  # Folder source — uncomment and set the path
  # - name: "docs"
  #   type: folder
  #   path: "/path/to/your/documents"
  #   extensions: [".pdf", ".txt", ".md"]
  #   collection: "{name}_docs"
  #   result_type: "document"
  #   search_weight: 0.5

search:
  relevance_min_score: 0.34
  cross_encoder_model: "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
  program_synonyms: {{}}
  module_synonyms: {{}}
"""
    dest.write_text(template, encoding="utf-8")
    log.info("Created config: %s", dest)
    log.info("Edit the sources section, then run:")
    log.info("  python cli.py ingest %s", dest)
    log.info("  python cli.py serve  %s", dest)


def cmd_eval(args):
    """Run evaluation queries against the live API."""
    test_file = Path(args.test_file) if args.test_file else Path("scripts/test_queries.json")
    if not test_file.exists():
        log.error("Test file not found: %s", test_file)
        sys.exit(1)

    # Import and run the eval script
    sys.path.insert(0, str(Path(__file__).parent / "scripts"))
    try:
        import eval as eval_module
        passed, total = eval_module.run_eval(args.url, args.top_k)
        sys.exit(0 if passed == total else 1)
    except ImportError:
        log.error("Could not import eval script from scripts/eval.py")
        sys.exit(1)


def cmd_list(args):
    """List sources defined in a config."""
    import yaml
    config_path = Path(args.config)
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    sources = config.get("sources", [])
    print(f"\nProject: {config['name']}")
    print(f"Sources ({len(sources)}):\n")
    for s in sources:
        print(f"  [{s.get('type','?')}] {s.get('name','?')}  →  {s.get('collection','?')}")
        if s.get("url"):
            print(f"         url: {s['url']}")
        if s.get("path"):
            print(f"         path: {s['path']}")
        print(f"         weight: {s.get('search_weight', 0):.0%}")
    print()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python cli.py",
        description="Generic Search-as-a-Service Framework",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Run the ingestion pipeline")
    p_ingest.add_argument("config", help="Path to YAML config file")
    p_ingest.add_argument("--source", action="append", metavar="NAME",
                          help="Only ingest this source (repeat for multiple)")

    # serve
    p_serve = sub.add_parser("serve", help="Start the search API server")
    p_serve.add_argument("config", help="Path to YAML config file")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)

    # new
    p_new = sub.add_parser("new", help="Create a new config from template")
    p_new.add_argument("name", help="Project name (used as config filename)")
    p_new.add_argument("--force", action="store_true", help="Overwrite existing config")

    # eval
    p_eval = sub.add_parser("eval", help="Run evaluation queries against the API")
    p_eval.add_argument("config", help="Path to YAML config file (for project name)")
    p_eval.add_argument("--url", default="http://localhost:8000", help="API base URL")
    p_eval.add_argument("--top-k", type=int, default=3)
    p_eval.add_argument("--test-file", default=None, help="Path to test_queries.json")

    # list
    p_list = sub.add_parser("list", help="List sources in a config")
    p_list.add_argument("config", help="Path to YAML config file")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

COMMANDS = {
    "ingest": cmd_ingest,
    "serve": cmd_serve,
    "new": cmd_new,
    "eval": cmd_eval,
    "list": cmd_list,
}


def main():
    parser = build_parser()
    args = parser.parse_args()
    COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
