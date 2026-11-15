"""WWW supplied-region QA commands. Public inference and private scoring are separate."""
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import sys

from ..io import read_json
from .cli import TerminalProgress
from .config import load_config


def module(name):
    return importlib.import_module(".2026-11-15_www_" + name, __package__)


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="action", required=True)
    imp = commands.add_parser("archive-import", help="Validate/copy the shared public archive package; CPU")
    imp.add_argument("--package", required=True)
    index = commands.add_parser("archive-index", help="Index actual source text/images with the existing Jina worker")
    index.add_argument("--corpus", required=True)
    index.add_argument("--config", required=True)
    index.add_argument("--text-chars", type=int, default=600)
    run = commands.add_parser("run", help="Execute the exact public cohort in three comparable conditions")
    run.add_argument("--questions", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--archive-index")
    run.add_argument("--allow-empty-archive", action="store_true")
    run.add_argument("--conditions", nargs="+", choices=module("runner").CONDITIONS, default=module("runner").CONDITIONS)
    run.add_argument("--protocol", help="Optional frozen JSON evidence limits, not model settings")
    run.add_argument("--resume-from")
    demo = commands.add_parser("demo", help="Clearly synthetic CPU fixture; no real model execution")
    for command in (imp, index, run, demo):
        command.add_argument("--out", required=True)
        command.add_argument("--quiet", action="store_true")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    progress = TerminalProgress(quiet=args.quiet)
    try:
        progress("command_started", task="www " + args.action, out=args.out)
        with contextlib.redirect_stdout(io.StringIO() if args.quiet else sys.stderr):
            if args.action == "archive-import":
                result = module("archive").import_archive(args.package, args.out)
            elif args.action == "archive-index":
                result = module("archive").index_archive(args.corpus, load_config(args.config), args.out,
                    text_chars=args.text_chars, progress=progress)
            elif args.action == "run":
                result = module("runner").run(args.questions, load_config(args.config), args.out,
                    archive_index=args.archive_index, allow_empty_archive=args.allow_empty_archive, conditions=args.conditions,
                    protocol=read_json(args.protocol) if args.protocol else None, resume_from=args.resume_from, progress=progress)
            else:
                result = module("demo").demo(args.out, progress=progress)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 1 if result.get("status") == "completed_with_errors" else 0
    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
        print(f"agentic www: {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        progress.close()
