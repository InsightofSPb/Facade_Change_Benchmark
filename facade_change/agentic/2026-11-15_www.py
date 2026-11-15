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
    inventory = commands.add_parser("inventory", help="CPU inventory of all canonical tracks/splits for archive research")
    inventory.add_argument("--histories", required=True)
    inventory.add_argument("--candidates")
    inventory.add_argument("--exposure-audit")
    inventory.add_argument("--addresses", help="Optional sourced address metadata; never infer an address from an ID")
    batch = commands.add_parser("batch", help="Select a deterministic batch before model outcomes")
    batch.add_argument("--candidates", required=True)
    batch.add_argument("--inventory", required=True)
    batch.add_argument("--size", type=int, default=100)
    batch.add_argument("--per-building", type=int, default=12)
    batch.add_argument("--seed", default="2026-11-15-www-v1")
    batch.add_argument("--max-overlap", type=float, default=.5)
    batch.add_argument("--cohort", choices=("development", "evaluation"), default="development")
    handoff = commands.add_parser("handoff", help="Import all 16 PRIVATE text judgments, with localization pending")
    handoff.add_argument("--bundle", required=True)
    handoff.add_argument("--candidates", help="Validate owner candidate bytes; omit only for portable handoff inspection")
    public = commands.add_parser("public", help="Export selected candidate questions for explicitly unscored inference")
    public.add_argument("--candidates", required=True)
    public.add_argument("--selection", required=True)
    public.add_argument("--limit", type=int)
    public.add_argument("--knowledge-cutoff", help="Explicit new freeze, e.g. for later archive collection; never backdate")
    review = commands.add_parser("review-page", help="Private static editor; exports explicit qa review decisions")
    review.add_argument("--candidates", required=True)
    review.add_argument("--selection", required=True)
    review.add_argument("--handoff")
    review.add_argument("--previous")
    plan = commands.add_parser("plan", help="Freeze approved cohort/strata/model contract before model execution")
    plan.add_argument("--questions", required=True)
    plan.add_argument("--answers", required=True)
    plan.add_argument("--inventory", required=True)
    plan.add_argument("--config", required=True)
    plan.add_argument("--archive-index")
    plan.add_argument("--allow-empty-archive", action="store_true")
    plan.add_argument("--protocol")
    score = commands.add_parser("score", help="Offline primary metrics, qa localization, known-support and blind review")
    score.add_argument("--questions", required=True)
    score.add_argument("--answers", required=True)
    score.add_argument("--run", required=True)
    score.add_argument("--plan", required=True)
    for command in (imp, index, run, demo, inventory, batch, handoff, public, review, plan, score):
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
            elif args.action == "inventory":
                result = module("curation").inventory(args.histories, args.out, candidates_path=args.candidates,
                    exposure_path=args.exposure_audit, addresses_path=args.addresses)
            elif args.action == "batch":
                result = module("curation").batch(args.candidates, args.inventory, args.out, size=args.size,
                    per_building=args.per_building, seed=args.seed, max_overlap=args.max_overlap, cohort=args.cohort)
            elif args.action == "handoff":
                result = module("curation").handoff(args.bundle, args.out, args.candidates)
            elif args.action == "public":
                result = module("curation").public_candidates(args.candidates, args.selection, args.out,
                    limit=args.limit, knowledge_cutoff=args.knowledge_cutoff)
            elif args.action == "review-page":
                result = module("curation").review_page(args.candidates, args.selection, args.out,
                    handoff_path=args.handoff, previous=args.previous)
            elif args.action == "plan":
                result = module("evaluation").plan(args.questions, args.answers, args.inventory, load_config(args.config), args.out,
                    archive_index=args.archive_index, allow_empty_archive=args.allow_empty_archive,
                    protocol=read_json(args.protocol) if args.protocol else None)
            elif args.action == "score":
                result = module("evaluation").score(args.questions, args.answers, args.run, args.plan, args.out)
            else:
                result = module("demo").demo(args.out, progress=progress)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 1 if result.get("status") == "completed_with_errors" else 0
    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
        print(f"agentic www: {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        progress.close()
