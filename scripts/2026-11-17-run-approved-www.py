#!/usr/bin/env python3
"""Apply existing human decisions, freeze A/B/C, then run with visible subprocess logs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def read(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_decisions():
    filename = "2026-11-16-www-assisted-decisions.private.json"
    paths = [Path.home() / "Downloads" / filename, Path.home() / "Загрузки" / filename]
    paths.extend(Path("/mnt/c/Users").glob("*/Downloads/" + filename))
    found = sorted({p.resolve() for p in paths if p.is_file()})
    if not found:
        raise ValueError("Не найден экспорт решений в Downloads. Передайте --decisions /абсолютный/путь/к/JSON")
    if len({file_sha(p) for p in found}) != 1:
        raise ValueError("Найдены разные экспорты решений; укажите --decisions явно: " + ", ".join(map(str, found)))
    return found[0]


def command(repo, *arguments, allow_errors=False):
    argv = [sys.executable, "-u", "-m", "facade_change.agentic", *map(str, arguments)]
    print("\nЗапуск: " + shlex.join(argv), flush=True)
    result = subprocess.run(argv, cwd=repo, check=False)
    if result.returncode not in ((0, 1) if allow_errors else (0,)):
        raise RuntimeError(f"Команда завершилась с кодом {result.returncode}; следующие этапы не запущены")
    return result.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--candidates", type=Path, default=repo / "runs/2026-11-15-www-candidates-all-001/candidates.json")
    parser.add_argument("--corpus", type=Path, default=repo / "runs/2026-11-16-www-home-v41-001/archive/corpus.json")
    parser.add_argument("--inventory", type=Path, default=repo / "runs/2026-11-16-www-home-v41-001/review/inventory/inventory.json")
    parser.add_argument("--config", type=Path, default=repo / "configs/2026-11-15-www-archive-20mp-001.local.json")
    parser.add_argument("--expected-approved", type=int, default=103)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepared-from", type=Path, help="Reuse unchanged public exports and archive index from a prior smoke; freeze new plans")
    parser.add_argument("--prepare-only", action="store_true", help="Validate/export the human decisions on CPU, without any model execution")
    parser.add_argument("--smoke-only", action="store_true", help="Run one approved case in A/C/B and print its offline metrics, then stop")
    parser.add_argument("--continue", dest="continue_prepared", action="store_true", help="Continue the successful smoke folder with all A, then all C, then all B")
    args = parser.parse_args(argv)
    if args.prepared_from and args.continue_prepared:
        raise ValueError("--prepared-from создаёт новый запуск; для его продолжения используйте только --continue")
    inputs = args.out / "inputs"
    frozen_decisions = inputs / "2026-11-17-decisions.private.json"
    prior_decisions = args.prepared_from / "inputs/2026-11-17-decisions.private.json" if args.prepared_from else None
    decisions = (args.decisions or prior_decisions or (frozen_decisions if args.continue_prepared else find_decisions())).resolve()
    for path in (decisions, args.candidates, args.corpus, args.inventory, args.config):
        if not path.is_file():
            raise ValueError(f"Отсутствует входной файл: {path}")
    if args.out.exists() and not args.continue_prepared:
        raise ValueError(f"Каталог уже существует: {args.out}; укажите новый --out")
    if args.continue_prepared and (args.out / "inference").exists():
        raise ValueError("Основной запуск в этом каталоге уже существует; не перезаписываем его")
    saved = read(decisions)
    if saved.get("candidate_sha256") != file_sha(args.candidates):
        raise ValueError("Экспорт решений относится к другим исходным candidates.json; решения не перенесены")
    counts = Counter(row["status"] for row in saved["decisions"])
    if counts["approved"] != args.expected_approved:
        raise ValueError(f"Ожидалось {args.expected_approved} approved, найдено {counts['approved']}; запуск остановлен")
    approved = {r["question_id"] for r in saved["decisions"] if r["status"] == "approved"}
    if len(approved) != args.expected_approved:
        raise ValueError("Повторяющиеся question_id в решениях")
    candidates = read(args.candidates)
    selected = [c for c in candidates["candidates"] if c["question_id"] in approved]
    if len(selected) != len(approved):
        raise ValueError("Подтверждённые вопросы отсутствуют в исходном manifest")
    inventory = read(args.inventory)
    if inventory.get("qa_policy", {}).get("id") != "rgb-qa-2026-11-16-v2":
        raise ValueError("Для всей подтверждённой выборки нужен существующий inventory RGB QA v2")
    identity_args = []
    identity = inventory.get("identity_source")
    if identity:
        identity_path = Path(identity["path"])
        if not identity_path.is_file() or file_sha(identity_path) != identity["sha256"]:
            raise ValueError("Исходный identity map отсутствует или отличается от inventory")
        identity_args = ["--identity-map", identity_path]
    views = {(v["building_id"], v["view_id"]): v for v in inventory["views"]}
    for case in selected:
        if not views.get((case["building_id"], case["view_id"]), {}).get("eligible"):
            raise ValueError(f"Непригодный или отсутствующий вид в inventory: {case['question_id']}")
        for observation in case["observations"]:
            for key in ("rgb_path", "support_path"):
                if not Path(observation[key]).is_file():
                    raise ValueError(f"Отсутствует ресурс: {observation[key]}")
    print(f"Решения: {decisions}\nСтатусы: {dict(counts)}\n"
          f"Основной запуск: {len(approved)} вопросов × 3 режима = {3 * len(approved)} ответов", flush=True)
    questions = args.out / "public/questions.json"
    answers = args.out / "public/answers.private.json"
    order = ["photos_only", "bounded_agentic_retrieval", "fixed_retrieval"]
    manifest_path = inputs / "2026-11-17-launch-inputs.private.json"
    reused_files = {}
    if args.prepared_from:
        previous_manifest = args.prepared_from / "inputs/2026-11-17-launch-inputs.private.json"
        recorded = read(previous_manifest)
        for path in (args.candidates, args.corpus, args.inventory, args.config):
            if recorded.get(str(path.resolve())) != file_sha(path):
                raise ValueError(f"Вход отличается от подготовленного запуска: {path}")
        for path, expected in recorded.items():
            if not Path(path).is_file() or file_sha(path) != expected:
                raise ValueError(f"Подготовленный вход изменился: {path}")
        if file_sha(decisions) != file_sha(prior_decisions):
            raise ValueError("Решения отличаются от подготовленного запуска")
        for folder, plan_folder in (("public", "plan"), ("smoke-public", "smoke-plan")):
            plan_path = args.prepared_from / plan_folder / "plan.private.json"
            previous_plan = read(plan_path)
            for name, key in (("questions.json", "questions_sha256"), ("answers.private.json", "answers_sha256")):
                path = args.prepared_from / folder / name
                if file_sha(path) != previous_plan[key]:
                    raise ValueError(f"Подготовленный экспорт изменился: {path}")
                reused_files[str(path.resolve())] = file_sha(path)
            path = args.prepared_from / "archive-index/index.json"
            if file_sha(path) != previous_plan["execution_contract"]["archive_index_sha256"]:
                raise ValueError("Подготовленный индекс не соответствует исходному плану")
            reused_files[str(path.resolve())] = file_sha(path)
            reused_files[str(plan_path.resolve())] = file_sha(plan_path)
        reused_files[str(previous_manifest.resolve())] = file_sha(previous_manifest)
    if not args.continue_prepared:
        args.out.mkdir(parents=True)
        inputs.mkdir()
        shutil.copyfile(decisions, frozen_decisions)
        if args.prepared_from:
            for folder in ("public", "smoke-public"):
                shutil.copytree(args.prepared_from / folder, args.out / folder)
            (args.out / "archive-index").mkdir()
            shutil.copyfile(args.prepared_from / "archive-index/index.json", args.out / "archive-index/index.json")
            print("Готовые экспорты и индекс переиспользованы без инференса; планы будут созданы для текущего кода.", flush=True)
        else:
            command(repo, "qa", "review", "--candidates", args.candidates, "--decisions", frozen_decisions,
                    "--reviewed-gallery-only", "--out", args.out / "review")
            command(repo, "qa", "export", "--review", args.out / "review/reviewed.json", "--out", args.out / "public")
        sources = [args.candidates, args.corpus, args.inventory, args.config, frozen_decisions, questions, answers]
        if identity:
            sources.append(identity_path)
        if args.prepared_from:
            sources.extend(args.out / name for name in ("archive-index/index.json", "smoke-public/questions.json", "smoke-public/answers.private.json"))
        manifest_path.write_text(json.dumps({**reused_files, **{str(p.resolve()): file_sha(p) for p in sources}}, indent=2) + "\n", encoding="utf-8")
    else:
        manifest = read(manifest_path)
        if any(str(p.resolve()) not in manifest for p in (args.candidates, args.corpus, args.inventory, args.config)):
            raise ValueError("Для продолжения используйте те же входные пути, что и для smoke")
        for path, expected in manifest.items():
            if not Path(path).is_file() or file_sha(path) != expected:
                raise ValueError(f"Вход изменился после smoke: {path}; основной запуск не начат")
        contract = read(args.out / "plan/plan.private.json")["execution_contract"]
        for name, expected in contract["implementation_sha256"].items():
            if file_sha(repo / "facade_change/agentic" / name) != expected:
                raise ValueError(f"Код изменился после smoke: {name}; нужен новый согласованный запуск")
        if read(args.out / "smoke-inference/summary.json")["outcome_counts"]["error"]:
            raise ValueError("Smoke содержит технические ошибки; основной запуск не начат")
    if {q["question_id"] for q in read(questions)["questions"]} != approved:
        raise RuntimeError("Экспорт изменил состав approved; модели не запущены")
    if file_sha(decisions) != file_sha(frozen_decisions):
        raise RuntimeError("Исходный экспорт решений изменился во время подготовки")
    if args.prepare_only:
        print("CPU-подготовка завершена; модели не запускались.", flush=True)
        return 0
    index = args.out / "archive-index/index.json"
    if not args.continue_prepared:
        if not args.prepared_from:
            command(repo, "www", "archive-index", "--corpus", args.corpus, "--config", args.config,
                    "--out", args.out / "archive-index")
        command(repo, "www", "plan", "--questions", questions, "--answers", answers, "--inventory", args.inventory,
                "--config", args.config, "--archive-index", index, "--include-reviewed", "--conditions", *order,
                "--condition-major", *identity_args, "--out", args.out / "plan")
        labels = {r["question_id"]: r["reviewed_labels"] for r in read(answers)["cases"]}
        changes = [q for q in read(questions)["questions"] if q["task"] == "temporal_change"
                   and labels[q["question_id"]]["assessment"] == "confirmed_change"]
        if not changes:
            raise ValueError("Нет подтверждённого temporal_change для полного smoke с локализацией")
        records = read(args.corpus)["records"]
        covered = [q for q in changes if any(r["identity_status"] == "verified" and r["building_id"] == q["building_id"]
                   and r["view_id"] in (None, q["view_id"]) for r in records)]
        smoke_id = (covered or changes)[0]["question_id"]
        print("\nE2E smoke: один существующий approved вопрос " + smoke_id, flush=True)
        if args.prepared_from:
            if [q["question_id"] for q in read(args.out / "smoke-public/questions.json")["questions"]] != [smoke_id]:
                raise ValueError("Повтор smoke должен использовать тот же заранее выбранный вопрос")
        else:
            command(repo, "qa", "export", "--review", args.out / "review/reviewed.json", "--question-id", smoke_id,
                    "--out", args.out / "smoke-public")
        smoke_questions = args.out / "smoke-public/questions.json"
        smoke_answers = args.out / "smoke-public/answers.private.json"
        command(repo, "www", "plan", "--questions", smoke_questions, "--answers", smoke_answers,
                "--inventory", args.inventory, "--config", args.config, "--archive-index", index,
                "--include-reviewed", "--conditions", *order, "--condition-major", *identity_args, "--out", args.out / "smoke-plan")
        smoke_status = command(repo, "www", "run", "--questions", smoke_questions, "--config", args.config,
                               "--archive-index", index, "--conditions", *order, "--condition-major", *identity_args,
                               "--out", args.out / "smoke-inference", allow_errors=True)
        command(repo, "www", "score", "--questions", smoke_questions, "--answers", smoke_answers,
                "--run", args.out / "smoke-inference", "--plan", args.out / "smoke-plan/plan.private.json",
                "--out", args.out / "smoke-score")
        report = read(args.out / "smoke-score/metrics.private.json")
        for condition in order:
            row = next(r for r in report["cases"] if r["condition"] == condition)
            print(json.dumps({"mode": condition, "status": row["status"], "decision": row["decision"],
                              "structured_correct": row["correct"], "error": row["error"]}, ensure_ascii=False), flush=True)
        print(f"Метрики smoke: {args.out / 'smoke-score/metrics.private.json'}", flush=True)
        if smoke_status:
            raise RuntimeError("E2E smoke содержит технические ошибки; метрики сохранены, основной запуск не начат")
        if args.smoke_only:
            print("E2E smoke завершён. Для основного прохода A → C → B повторите команду с --continue вместо --smoke-only.", flush=True)
            return 0
    print("\nНачинается основной проход: весь A, затем весь C, затем весь B.", flush=True)
    result = command(repo, "www", "run", "--questions", questions, "--config", args.config, "--archive-index", index,
                     "--conditions", *order, "--condition-major", *identity_args,
                     "--out", args.out / "inference", allow_errors=True)
    command(repo, "www", "score", "--questions", questions, "--answers", answers,
            "--run", args.out / "inference", "--plan", args.out / "plan/plan.private.json", "--out", args.out / "score")
    print(f"\nРезультаты: {args.out / 'inference/gallery.html'}\nМетрики: {args.out / 'score/metrics.private.json'}", flush=True)
    return result


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"Остановка: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
