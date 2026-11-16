# WWW RGB QA: новая подготовка выборки, 2026-11-16

Политика `rgb-qa-2026-11-16-v2` реализует явно согласованное владельцем изменение
**нового RGB QA-эксперимента**. Это подготовка для ручной проверки, не approval
и не результат сравнения моделей. Исходное сегментационное разбиение, старые
эксперименты, references, prompt, scoring и GPU-бюджеты не изменяются.

Включение явное: `www inventory` / `www review-package --qa-policy
rgb-qa-2026-11-16-v2`. `www batch` наследует политику сохранённого inventory;
другая политика требует нового inventory. Без флага остаётся прежняя
`segmentation-splits-2026-11-15-v1` с её ограничениями. Полный контракт новой
политики записан в `qa-policy.json`, inventory, selection и frozen plan; при
`www public` её ID/SHA входят в публичный manifest, без приватных support labels.

## Split, QA exposure и физическое здание

TRAIN, VAL и TEST допустимы для нового review: Qwen/Jina заморожены, предсказания
обученной нами сегментационной модели не поступают в QA. `split` и
`original_split`, прежние `original_*`/`legacy_*` exposure flags и исходный audit
сохраняются. `segmentation_usage` отделяет назначение split от документированного
использования: принадлежность TRAIN сама по себе не доказывает фактическое обучение.

QA exposure учитывается отдельно:

- `--exposed-questions` (повторяемый) фиксирует использованные вопросы и экспозицию
  их физических зданий; ID и перекрывающиеся события исключаются из нового batch;
- scoped `www_exposure_audit` должен содержать `exposure_domain: "rgb_qa"`,
  `qa_policy: "rgb-qa-2026-11-16-v2"`, прежние `protocol_source:{path,sha256}`,
  `protocol_sha256`, `buildings`. Для `prior_exposure: "none_found"` нужны reviewer,
  реальные `audit_sources:[{path,sha256}]` и проверенные хеши. Ранее использованное
  для разработки QA здание отмечается `prior_exposure: "used_for_qa_development"`;
- legacy audit сохраняется, но не превращается автоматически в доказательство
  отсутствия QA exposure. Неизвестный статус — `unknown`, cohort — `unassigned`:
  review допустим, `evaluation_eligible=false`. Freeze для оценки требует отдельного
  ручного exposure/cohort review; неизвестное не называется отсутствием утечки.

Можно записать отдельно документированное `segmentation_usage` в building-строке
audit. Отсутствующее значение остаётся `not_audited`. Известная QA-экспозиция
распространяется на всю подтверждённую физическую группу, включая alias без
tracks в текущем histories. Другие независимые события такого здания могут попасть
в новый review, но здание остаётся development, а не untouched evaluation.

Voskova_10/Voskova_12 и baskov_32/baskov_34 остаются pending, пока человек не
проверит identity. Pending-группа исключается из отбора, но остаётся в inventory
и pending review. Только подтверждённое `same_building` объединяет виды.
Для него `segmentation_split_overlap` — сохранённый факт, а
`qa_development_evaluation_overlap` — отдельная проверка QA-разделения.
Конфликт QA development/evaluation не даёт независимую evaluation.
Одинаковые исходные RGB у разных неотождествлённых физических зданий по-прежнему
вызывают исключение. При retrieval, freeze и scoring используется тот же
проверенный identity map; ограничения `view_ids` сохраняются.

## Отбор и шесть локальных изображений

Целевой размер — 100, максимум — 12 на физическое здание. Каждый детерминированный
проход выбирает **один** вопрос каждого доступного здания. Внутри здания сначала
предпочитаются менее представленные типы, с мягким ориентиром два вопроса типа.
Другие независимые события можно выбирать сверх двух. Отсутствие типа не блокирует
остальные. 40/30/20/10 показываются только как справочная смесь; это не квоты и не
метки. Генератор кандидатов, их текст и candidate families не изменены; новые
stability/insufficient cases для заполнения чисел не создаются.

Сохраняется имеющееся подавление перекрывающихся предложений одного интервала
на общей anchor/view решётке. Подтверждение одинакового физического события
между разными видами/интервалами остаётся ручной задачей; геометрический heuristic
не объявляется доказательством независимости событий.

Перед отбором проверяется текущий `max_local_images=6`. Все обязательные даты
должны присутствовать и помещаться в бюджет. Для earliest нельзя скрыто урезать
набор наблюдений: более шести даёт
`earliest_observation_set_exceeds_image_cap_scope_revision_required`.
Такой вопрос перечисляется отдельно; изменение его области требует новой
явно проверяемой revision. Исходная длинная история остаётся источником вопросов
по отдельным интервалам. Для совместимых вопросов записаны выбранные observation
IDs и пропущенные **необязательные** IDs, в точности по текущей логике runner.
GPU/token/image budgets не увеличиваются.

Отчёт:

- `summary.json`: requested/selected/shortfall, physical buildings, типы,
  original splits, QA exposure/cohort, причины исключений и image compatibility;
- `batch/composition.private.json` и `.csv`: состав по вопросам и зданиям,
  справочное отклонение от смеси — в summary; `target_shortfalls={}` означает,
  что категорийных квот нет, а не что вся выборка набрана;
- `batch/selection.private.json`: точные selected/excluded IDs и причины;
- `batch/image-budget.private.csv`: совместимость **всех** кандидатов;
- `inventory/inventory.json`: исходные IDs, splits, exposure, alias и eligibility;
- `index.html`, `review/review.html`: прежний приватный редактор с dated RGB,
  source/date metadata, архивом и редактируемыми предложениями, approvals=0.

## Архив: источники, поиск и полезность

В новой политике `--archive-corpus` необязателен. Наличие архива у каждого здания
не является фильтром отбора. `archive-coverage.private.json` отдельно показывает
source presence, совместимые evidence IDs и human-confirmed useful evidence.
Наличие источника не доказывает полезность; отсутствие источника в переданном
корпусе не означает, что поиск проводился. Без журнала статус `not_documented`.

Необязательный `--archive-coverage-review` читает **приватный** журнал:

```json
{
  "kind": "www_archive_coverage_review",
  "access": "private_evaluation",
  "corpus_sha256": "<SHA фактически переданного corpus.json>",
  "buildings": [],
  "support": []
}
```

Building-строка: `building_id`, `search_status` (`not_documented`, `not_searched`,
`searched_no_sources`, `searched_with_sources`), `reviewer`, `notes`,
`audit_sources:[{path,sha256}]`. Утверждение о проведённом поиске требует источников
с проверенными хешами. Support-строка: `question_id`, `evidence_id`, `verdict`
(`useful`, `not_useful`, `unresolved`), `reviewer`, `notes`.
`useful` требует допустимых identity/view/date bindings точного корпуса.
Журнал описывает review, не меняет selection/answers и не передаётся модели.
Correspondence boxes приватной галереи по-прежнему отделены от change-event GT.
Supplement V4.1 не становится публичным корпусом и автоматически не утверждает
ни identity, ни mappings, ни полезность.

## Расходы без нового инференса

`www costs --run <saved-run> --out <new-directory>` читает сохранённые costs/calls
и execution contract; RGB не открывает, workers не запускает. `www audit` также
сохраняет этот отчёт. Выход: `costs.private.json`, `costs.private.csv`, summary и
gallery. A/B/C — `photos_only` / `fixed_retrieval` / `bounded_agentic_retrieval`.

Для каждого режима и каждой ожидаемой строки сохраняются Qwen/Jina отдельно:
input/output tokens, `requests`, фактические `model_calls`, elapsed seconds,
статусы, retries и cache hits. Поиски учитываются отдельно от вызовов Jina.
Технические попытки уже суммированы ModelPool: повторно их usage не складывается.
Дополнительные call artifacts включаются, даже если отсутствуют в ledger;
потерянный artifact использует записанный usage с явной отметкой неполноты.
Ошибки и отсутствующие outputs остаются в ожидаемом числе строк.

Кэшированный `source_usage` и перенесённый `source_costs` сохраняются отдельно,
не списываются повторно. Resource-verification seconds/bytes выделены отдельно.
Неизмеренные значения помечаются, totals трактуются как **наблюдённая часть**,
а не полная стоимость. Отдельное построение archive index не приписывается
каждому A/B/C запуску: его расходы остаются в собственном index run. Меньшая
стоимость сама по себе не означает преимущество метода или factual correctness.

## WSL: foreground, сохранение original-16 и новая выборка

Ниже команда для all-candidates/V4.1 из предыдущего этапа. Наличие этих файлов
в WSL здесь не проверено. Они должны относиться к одной histories revision,
содержать TRAIN/VAL/TEST и иметь cutoff фактического публичного V4.1. Приватный
ZIP supplement не заменяет `corpus.json`. Каталог результата должен быть новым.
Исходный handoff **не переносится** на новые candidate bytes; original-16 остаются
на своей отдельной странице и в своих файлах. Использованные вопросы передаются
через `--exposed-questions`, другие использованные manifests — повторением флага.

```bash
cd /home/sasha/Facade_Change_Benchmark &&
git pull --ff-only &&
set -o pipefail &&
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -u -m facade_change.agentic www review-package \
  --qa-policy rgb-qa-2026-11-16-v2 --size 100 \
  --candidates /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-candidates-all-v41-001/candidates.json \
  --histories /home/sasha/Facade_Change_Benchmark/data/histories \
  --archive-corpus /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-archive-v41-001/corpus.json \
  --exposed-questions /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-public-16-001/questions.json \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-rgb-review-v2-001 \
  2>&1 | tee /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-rgb-review-v2-001.console.log
```

Если all-candidates ещё нет, ниже полный CPU workflow с **подтверждёнными ранее
путями V3.1**, без предположения о готовности V4.1. Это отдельная V3.1-подготовка:
она не выдаётся за V4.1. Для V4.1 нужны фактический публичный корпус с assets,
его cutoff и новые output paths; уже готовый совместимый manifest можно
переиспользовать вместо `qa generate`. `available_at` в новой revision
консервативно равен freeze, старые даты/bytes не переписываются.

```bash
cd /home/sasha/Facade_Change_Benchmark &&
git pull --ff-only &&
set -o pipefail &&
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -u - <<'PY' 2>&1 | tee /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-rgb-prepare-v31-001.console.log
import json
from pathlib import Path
import subprocess
import sys

root = Path('/home/sasha/Facade_Change_Benchmark')
corpus = root / 'runs/2026-11-15-www-archive-v31-001/corpus.json'
cutoff = json.loads(corpus.read_text())['knowledge_cutoff']
def run(*args):
    print('CPU:', ' '.join(map(str, args)), flush=True)
    subprocess.run([sys.executable, '-u', '-m', 'facade_change.agentic', *map(str, args)], check=True)
run('www', 'review-page',
    '--candidates', root / 'runs/2026-11-15-facade-qa-candidates-001/candidates.json',
    '--selection', root / 'runs/2026-11-15-www-handoff-001/selection.private.json',
    '--handoff', root / 'runs/2026-11-15-www-handoff-001/handoff.private.json',
    '--archive-corpus', corpus, '--out', root / 'runs/2026-11-16-www-rgb-original-16-001')
run('qa', 'generate', '--histories', root / 'data/histories',
    '--annotation-targets', root / 'runs/2026-10-25-history-targets-001/index.json',
    '--knowledge-cutoff', cutoff, '--task-mode', 'retrospective', '--available-at', cutoff,
    '--splits', 'train', 'val', 'test', '--out', root / 'runs/2026-11-16-www-rgb-candidates-v31-001')
run('www', 'review-package', '--qa-policy', 'rgb-qa-2026-11-16-v2', '--size', '100',
    '--candidates', root / 'runs/2026-11-16-www-rgb-candidates-v31-001/candidates.json',
    '--histories', root / 'data/histories', '--archive-corpus', corpus,
    '--exposed-questions', root / 'runs/2026-11-16-www-public-16-001/questions.json',
    '--out', root / 'runs/2026-11-16-www-rgb-review-v31-001')
PY
```

Для применения ручного identity/exposure review создайте **новый** output с
`--identity-map` и `--exposure-audit`; pending template не является решением.
Приватную галерею можно добавить прежними `--private-audit` и
`--private-audit-supplement` с фактическими путями к
`2026-11-16-www-v4-private-region-review.zip` и
`2026-11-16-v4.1-correction-private-handoff.zip`.

Сохранённый реальный run на 16 вопросах можно разобрать отдельно, без инференса:

```bash
cd /home/sasha/Facade_Change_Benchmark &&
set -o pipefail &&
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -u -m facade_change.agentic www costs \
  --run /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-models-16-001 \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-role-costs-16-001 \
  2>&1 | tee /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-role-costs-16-001.console.log
```

## Выполненная CPU-проверка и границы результата

```bash
/workspace/work/facade-venv/bin/python -s -B -m unittest \
  tests.test_2026_11_16_www_rgb_policy \
  tests.test_2026_11_15_www_curation \
  tests.test_2026_11_16_www_review \
  tests.test_2026_11_16_www_audit -v
```

40 tests OK. Проверены новая/старая политики, TRAIN, deterministic building passes,
отсутствующая категория, soft/hard caps, длинная история и точное соответствие
image selection runner, pending/confirmed aliases и QA overlap, сохранение
handoff/candidate bytes, исключение экспонированных вопросов/перекрытий, private
archive usefulness и раздельные роли расходов с error/retry/cache/resume/missing.
Потерянные calls используют сохранённые role usage; неатрибутированные aggregate
tokens явно отмечаются как неполнота, а не объявляются нулевой стоимостью.

Сквозной CLI `www review-package` на новом синтетическом fixture:
**13/100 вопросов, 6 зданий, недобор 87**; TRAIN 10 / VAL 1 / TEST 2;
visible-change proposals 2 / stability 2 / insufficient 9 / earliest 0.
Все 13 имеют `qa_exposure=unknown`, `cohort=unassigned`, а не проверенную evaluation.
Из 16 кандидатов исключены один использованный вопрос и перекрывающееся событие,
один earliest с девятью обязательными датами. Восемь interval-вопросов длинной
истории сохранились. Это families исходных синтетических предложений, не approvals.
Сохранённое синтетическое human decision и входные bytes неизменны; approved=0.
Результат: `/workspace/work/2026-11-16-www-rgb-review-cpu-001`.

`www costs` разобрал **9 ранее сохранённых scripted** строк из
`/workspace/work/2026-11-16-www-archive-policy-cpu-001/inference`, без повторного
инференса. Qwen calls A/B/C = 3/3/6, Jina calls = 0/4/4, searches = 0/3/3;
неполные измерения этого fixture явно отмечены. Это проверка учёта, не фактическая
стоимость Qwen/Jina. Результат после проверки неполных totals:
`/workspace/work/2026-11-16-www-role-costs-cpu-002`.
Stdout обоих CLI — валидный JSON, stderr содержит видимый progress.

Полного WSL manifest и результатов реального run здесь нет. **Реальная выборка
из 100 вопросов не создана**, её состав/недобор ещё нужно получить в WSL. Сообщённые
владельцем 48 строк (44 completed / 3 error / 1 abstained) не переоценивались.
Исходные 16 решений и old candidate files не менялись; на WSL их SHA проверяет
отдельный `review-page`. Все 16 объектов решений, включая case 03, дополнительно
сверены с исходным coordination ZIP: совпадают с ранее сохранённым CPU handoff.
Проверены `git diff --check`, синтаксис четырёх Bash-блоков и встроенного Python
в этом документе. Ни GPU, ни веса, ни окружения не затрагивались.
