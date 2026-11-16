# CVPR: подготовка существующего inspection к реальному smoke — 2026-11-16

Рабочая ветка: `codex/2026-11-10-adaptive-visual-inspection`.
Начальный HEAD и remote проверены: `336964e5b3dcb9807b0dfd16f4814eb4d6191d1e`,
рабочее дерево было чистым. Закреплённая база остаётся
`9aa159c0489e3a4ee15a089b1e214ade8d980fc3`. Для выборочного сравнения прочитан
WWW `41ee86e694c72a9150b2c3b3054e9f7f735d721c`; merge/cherry-pick не выполнялись.

Сохраняются `pair / fixed_history / heuristic / agent`, два inspect-действия,
разрешённый stop, общий замороженный RGB-набор C/D и отдельный свежий judge.
Селектор входит в стоимость режима; его transcript и ответ отсутствуют у judge.
Исполнение с ошибкой остаётся error. Отказ модели остаётся abstained.
Все случаи, включая missing/error/abstained, остаются в оценке.

Основной endpoint: **agent − heuristic, building-macro event F1, K=1, IoU=0.25**.
Основная группа — физические здания с положительными GT-событиями. Стабильные
случаи внутри них дают FP. Stable-only здания остаются отдельной диагностикой.
IoU=0.5, P/R, отказы, ошибки и расходы дополнительны. `event_id` дедуплицируется
до агрегации существующим matcher; рамки разных регионов/ракурсов не объединяются.
В comparison.json явно перечислены `primary.positive_event_buildings` и
`stable_only_buildings`. Протокол действует внутри заданных зарегистрированных
участков; старые mask-difference метрики с reviewed-event метриками не смешиваются.

## Изменения и границы переноса

* `2026-11-16_inspection_reply.py`: один JSON-объект, внешние пробелы и один полностью
  закрытый JSON fence. Plain/fenced идут через один validator селектора/judge.
  `response-N.json` сохраняет исходный ответ; `reply-N.json` — точный текст,
  косметические преобразования, parsed object и validation error. Дубли ключей,
  NaN/Infinity/overflow, обрыв, текст вокруг, два объекта и нечисловые координаты
  отклоняются. `boxes=[]`, удаление лишних backticks и архивная схема WWW не перенесены.
* `inspection inventory`: CPU-таблица eligible/ineligible/pending, причины,
  числа до/после фильтров, распределения по физическим зданиям, split и числу дат.
  Принимает candidates.json без approvals либо публичный questions.json. Не читает
  predictions, не запускает модели и не проверяет наличие весов. Недоступные RGB
  дают pending; имеющиеся файлы всё равно проверяются по SHA. Проблема целостности
  самого history manifest является ошибкой команды, а не подтверждённым подсчётом.
* Детерминированный отбор по умолчанию: физические здания и их question IDs
  сортируются, затем берутся по одному ID из каждого здания по кругу до `--limit`
  (по умолчанию 30). Это верхняя граница, не обещание числа подходящих пар.
  `--ids` принимает JSON-массив явно выбранных eligible IDs в заданном порядке.
  Правило и порядок сохраняются до предсказаний. Модельные результаты не участвуют.
* `inspection prepare`: экспорт approved allowlist через прежний `qa.export`,
  проверка исходной review-chain прежним evaluator, независимые full/smoke exports
  и protocol SHA. Smoke — ровно два ID того же full cohort; по умолчанию первые два,
  либо `--smoke-ids` с явным JSON-массивом. Вопросы, wording, IDs, frame, RGB/support
  SHA, review state/revision, события, ответы и bbox сохраняются. Родительские файлы
  не меняются. Семантическая нормализация approved label запрещена.
* Физическая идентичность: перенесён узкий validator формата
  `www_physical_identity_review`. Известные предложения Voskova_10/Voskova_12 и
  baskov_32/baskov_34 по умолчанию **pending**, а не одобренные объединения.
  `--identity-map` связывает решение reviewer и файлы-источники по SHA. Для выбранных
  групп проверяются все исходные split в canonical manifest, включая невыбранные
  наблюдения. Pending, отсутствующий участник аудита и split conflict блокируют
  заморозку. При отсутствии затронутых групп mapping не нужен. Исходные building/view
  IDs и split не переписываются; физическая группировка применяется только к статистике
  и ownership событий. Никакой TEST/exposure-политики WWW не перенесено.
* `models.py`: только неперсистентный verifier из WWW. Первое полное SHA-чтение,
  привязка к resolved path, st_dev/inode/size/mtime_ns/ctime_ns/PID, отказ при изменении
  во время чтения. Persisted cache и model-visible images сохраняют полную проверку.
  `usage.resource_validation` содержит seconds/bytes_hashed/files_hashed/stat_cache_hits.
  Реальное ускорение на весах Qwen не измерено.
* `TerminalProgress` подключён к journal у preflight/run и CPU-подготовки. На stderr
  видны подготовка, case/policy, planner/judge request/return, действие, stop,
  invalid reply, повтор модели, error и итог. stdout остаётся JSON.
* `cost_summary.json` и `comparison.costs`: расходы по режимам/этапам, фактические
  вызовы/cache, повторы, presentations, токены, elapsed; shared preparation считается
  один раз на случай. Cache-source budget usage отделён от текущего usage. Неизвестные
  totals остаются null; observed_sum — частичная измеренная сумма, не полный расход.

Не перенесены WWW runner/retrieval, Jina, grayscale adapter, архивный date/citation
contract, разрешение TEST и весь WWW-стек. Qwen остаётся Qwen/Qwen3.5-4B,
revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, BF16/SDPA/greedy/local-files-only.
Работающая логика AnyChange, RGB-native crops, fresh judge и лимитов сохранена.

## Assessability без повторного рисования рамок

`inventory` создаёт приватные `gallery.html`, `inventory.private.json` с точным
разрешённым набором (даты, IDs, original/crop/support paths и SHA, геометрия),
а также `assessability-template.private.json`. Галерея показывает сохранённые RGB-кропы;
исходные native RGB для проверки детализации перечислены в inventory. Поля template
изначально pending. Это **не approval** и не декларация готовности выборки.

После обычного совместного WWW review используется его настоящий `reviewed.json`.
В новой копии template для каждого выбранного случая reviewer подтверждает только
assessability и границы допустимых свидетельств. Требуются:

```json
{
  "question_id": "EXACT_SELECTED_ID",
  "universe_sha256": "UNCHANGED_VALUE_FROM_TEMPLATE",
  "decision": "compatible",
  "review_source_sha256": "SHA_OF_EXISTING_REVIEWED_JSON",
  "review_revision": 1,
  "reviewer": "ACTUAL_REVIEWER",
  "rationale": "Почему исходный approved ответ совместим с этим точным RGB-набором",
  "basis": "narrow_human_scope_review",
  "sources": [{"path": "/ACTUAL/DOCUMENTED/CONFIRMATION", "sha256": "ACTUAL_SHA"}]
}
```

`review_revision` берётся из `reviewed.states[question_id].revision`; пример `1`
не является предположением о реальной revision. Верхнеуровневые kind, access и
selection_sha256 из template сохраняются. Если уже существует документированное
подтверждение **этого точного набора**, допустимо `basis=existing_documented_confirmation`
со ссылкой/SHA на него. Обычный WWW `question_scope=frozen_collection`, свободная заметка
без проверяемого охвата либо сам факт approved не заменяют подтверждение.

При incompatible/pending команда останавливается до экспорта. Исходный WWW-label
сохраняется: исключите/отложите ID новым allowlist и новой inventory/selection либо
получите отдельный подтверждённый CVPR review через прежний review workflow.
Не меняйте ответ или рамку программно. Повторное рисование неизменившихся рамок не нужно.

`prepare` создаёт публичный `admission.json`, содержащий только hashes и IDs;
protocol связывает его SHA. Preflight/run проверяют точный RGB universe и identity-map
через него. Приватный confirmation, selection diagnostics и answers не читаются run
и не попадают в evidence/prompt. `run` по-прежнему не принимает `--answers`.
Прежний явно заполненный протокол 2026-11-10 остаётся читаемым как аттестация владельца;
новая prepare-команда никогда не ставит её без указанной проверки.

## Что отсутствует в Cloud

Реальные `/home/sasha/...` здесь отсутствуют. Найдены только прежние процедурные
фикстуры. Поэтому число пригодных реальных пар/физических зданий и достижимость
20–30 **не установлены**. 8 491 кандидатов и новые 100 WWW-вопросов не считаются
CVPR cohort. Необходимы настоящие candidates/public questions, canonical manifest
с completed preparation и original RGB/support, approved reviewed.json и вся его
цепочка, narrow assessability confirmation, при затронутых aliases — reviewed identity-map,
реальный AnyChange source/checkpoint и действующий локальный config/model cache.
Веса не скачивались; реальные AnyChange/Qwen и GPU smoke не запускались.
CPU не подтверждает 16k/8 images на RTX 4060 Ti 16 GB.

## CPU-проверки реализации

Интерпретатор Cloud: `/workspace/work/facade-venv/bin/python` (существующая CPU-среда,
без установки Torch/весов). Успешные проверки:

```text
python -u -s -B -m unittest 'tests/test_2026-11-10_inspection.py' 'tests/test_2026-11-16_inspection.py' 'tests/test_2026-11-16_resource_verification.py' tests.test_2026_11_08_agentic_reasoning tests.test_2026_11_08_agentic_tool_regressions tests.test_2026_11_10_facade_questions tests.test_2026_11_08_agentic_cli tests.test_2026_11_08_agentic_models -v
117 tests: OK, без skips.

python -u -s -B -m unittest discover -s tests -p test_workflow.py -v
69 tests: OK, без skips; 47 повторяют inspection 24+16 и resource 7.
```

Итого **139 различных тестов**, а не 186. После последнего усиления проверки
неизменности selection отдельно повторены только 16 затронутых новых тестов: OK.
В них проходит полный синтетический путь inventory → reuse approved review →
full/smoke → C/D run → inspection evaluate, без чтения private answers в run.
Проверены plain/fenced parity без лишнего вызова, malformed retry, invalid schema,
ownership/frame/date/SHA, pending identity/assessability, split conflict, event dedup,
private isolation, видимые этапы и неизвестные расходы. `git diff --check` чистый.
`bash -n` проверил все 12 WSL-блоков; команды владельца в Cloud не исполнялись.

Во время разработки устранены две regression-проблемы: ранний admission скрывал
число/причины исключения исторических дат; галерея требовала локальные assets.
Последующие проверки восстановили исходную диагностику и используют штатный `_copy`.
Это журнал исправлений тестов, не результат реальной модели.

Отдельный CLI demo выполнен на чистом кодовом коммите
[`70b1b98f9f11af90b9e85197e5b37442900b5b85`](https://github.com/InsightofSPb/Facade_Change_Benchmark/commit/70b1b98f9f11af90b9e85197e5b37442900b5b85),
`source_snapshot.dirty=false`, каталог Cloud
`/workspace/work/2026-11-16-inspection-demo-001`. Analysis/evaluation completed;
запрограммированные heuristic F1=1/3 и agent F1=1 сохранились. В каждом режиме
4 completed, 1 abstained, 0 error, 0 missing. Положительные **синтетические** здания
основной метрики: b1/b2; stable-only: b3; b4 — unassessable. Это не измерение качества VLM.

CLI inventory на кандидатах этой фикстуры:
`/workspace/work/2026-11-16-inspection-inventory-001`, completed, 15 исходных кандидатов,
5 eligible / 10 ineligible, выбраны 5 случаев четырёх синтетических зданий.
Весь assessability template остался pending. На stderr появились этапы до итогового JSON.
Эти числа не переносятся на реальные данные. Полные CPU-логи сохранены вне Git в
`/workspace/work/2026-11-16-{focused-tests-003,workflow-tests,cohort-final-tests,inspection-demo-001,inspection-inventory-001}.log`.

## WSL: безопасное обновление и обнаружение входов

Каждый блок запускается отдельно в foreground. Оболочка блока не оставляет shell
variables/aliases. `set -C` и создание лога до `tee -a` запрещают перезапись старого
лога; выходные каталоги должны быть новыми. При существующем суффиксе `001` выберите
следующий свободный суффикс во всём соответствующем блоке. Старые результаты не удаляйте.

```bash
git -C /home/sasha/Facade_Change_Benchmark worktree list
git -C /home/sasha/Facade_Visual_Inspection status --short --branch
(
  set -euo pipefail
  cd /home/sasha/Facade_Visual_Inspection
  test "$(git branch --show-current)" = codex/2026-11-10-adaptive-visual-inspection
  test -z "$(git status --porcelain)"
  git fetch origin codex/2026-11-10-adaptive-visual-inspection
  git merge-base --is-ancestor HEAD origin/codex/2026-11-10-adaptive-visual-inspection
  git pull --ff-only origin codex/2026-11-10-adaptive-visual-inspection
  git rev-parse HEAD
)
```

Если worktree отсутствует или имеет другие изменения/ветку, этот блок остановится.
Сначала сохраните изменения и установите фактическое расположение CVPR worktree;
не применяйте reset, force или полный WWW merge.

Найдите входы по именам, затем проверьте их kind и происхождение. Prediction-файлы
для отбора не открывайте. Ниже известный путь histories проверяется, а не предполагается:

```bash
(
  set -euo pipefail
  set -C
  cd /home/sasha/Facade_Visual_Inspection
  mkdir -p runs
  : > runs/2026-11-16-cvpr-input-discovery-001.log
  rg --files --hidden --no-ignore /home/sasha/Facade_Change_Benchmark/data /home/sasha/Facade_Change_Benchmark/runs /home/sasha/Facade_Change_Benchmark/configs \
    | rg '(^|/)(candidates|questions|reviewed|manifest)\.json$|identity.*\.json$|local\.json$' \
    2>&1 | tee -a runs/2026-11-16-cvpr-input-discovery-001.log
  test -f /home/sasha/Facade_Change_Benchmark/data/2026-11-04-histories-s192-001/manifest.json
  test -f /home/sasha/Facade_Change_Benchmark/configs/2026-11-08-agentic-wsl-16k-001.local.json
)
```

В следующих блоках `/ОБЯЗАТЕЛЬНЫЙ/...` — **обязательные неизвестные входы**, которые
нужно заменить абсолютными найденными путями. Эти шаблоны не являются готовыми
путями. Источник inventory — candidates.json до review или публичный questions.json;
не reviewed.json, answers или predictions. Для найденных aliases добавьте
`--identity-map /ABSOLUTE/REVIEWED_IDENTITY.json`; пустой шаблон не заменяет review.

```bash
(
  set -euo pipefail
  set -C
  cd /home/sasha/Facade_Visual_Inspection
  test ! -e runs/2026-11-16-cvpr-inventory-001
  : > runs/2026-11-16-cvpr-inventory-001.log
  /home/sasha/miniconda3/envs/change-agentic-rag/bin/python -u -s -B -m facade_change.agentic inspection inventory \
    --source '/ОБЯЗАТЕЛЬНЫЙ/НАЙДЕННЫЙ/candidates.json' \
    --histories /home/sasha/Facade_Change_Benchmark/data/2026-11-04-histories-s192-001 \
    --available-at 'ОБЯЗАТЕЛЬНАЯ_ДАТА_ИЗ_ИСХОДНОГО_ЭКСПОРТА_ИЛИ_observation_date' \
    --limit 30 --out runs/2026-11-16-cvpr-inventory-001 \
    2>&1 | tee -a runs/2026-11-16-cvpr-inventory-001.log
)
```

`available-at` обязан совпадать с публичными наблюдениями. Не подставляйте текущую
дату по умолчанию. Из отчёта прочитайте counts, filters, distributions и причины;
при нехватке случаев условия не ослабляйте. Для собственного отбора сохраните новый
JSON-массив ID и повторите inventory в новом каталоге с `--ids /ACTUAL/ids.json`.
Зафиксированные ID можно совместно разметить один раз штатным WWW review.

После review скопируйте template в новый приватный файл и заполните только
подтверждение охвата свидетельств по описанному выше контракту. SHA и revision:

```bash
(
  set -euo pipefail
  set -C
  cd /home/sasha/Facade_Visual_Inspection
  : > runs/2026-11-16-cvpr-review-binding-001.log
  /home/sasha/miniconda3/envs/change-agentic-rag/bin/python -u -s -B - '/ОБЯЗАТЕЛЬНЫЙ/ОДОБРЕННЫЙ/reviewed.json' <<'PY' 2>&1 | tee -a runs/2026-11-16-cvpr-review-binding-001.log
import json
import sys
from facade_change.io import read_json, sha256
review = read_json(sys.argv[1])
print(json.dumps({"review_sha256": sha256(sys.argv[1]), "states": {
    key: {"status": row["status"], "revision": row["revision"]}
    for key, row in review["states"].items()}}, ensure_ascii=False, indent=2))
PY
  test ! -e runs/2026-11-16-assessability-review-001.private.json
  cp runs/2026-11-16-cvpr-inventory-001/assessability-template.private.json runs/2026-11-16-assessability-review-001.private.json
)
```

Следующий блок выполняйте после ручного подтверждения. Он создаст разные каталоги
`full` и `smoke`, каждый с `export/questions.json`, `export/answers.private.json`,
собственным admission и protocol. `selection.private.json` связывает источники и
выходные SHA. Это не повторный review рамок.

```bash
(
  set -euo pipefail
  set -C
  cd /home/sasha/Facade_Visual_Inspection
  test ! -e runs/2026-11-16-cvpr-cohort-001
  : > runs/2026-11-16-cvpr-cohort-001.log
  /home/sasha/miniconda3/envs/change-agentic-rag/bin/python -u -s -B -m facade_change.agentic inspection prepare \
    --review '/ОБЯЗАТЕЛЬНЫЙ/ОДОБРЕННЫЙ/reviewed.json' \
    --selection runs/2026-11-16-cvpr-inventory-001/selection.json \
    --assessment-review runs/2026-11-16-assessability-review-001.private.json \
    --out runs/2026-11-16-cvpr-cohort-001 \
    2>&1 | tee -a runs/2026-11-16-cvpr-cohort-001.log
)
```

## WSL: реальный config, preflight и отдельные GPU-запуски

CPU-подготовка разрешена во время WWW GPU-run. `check-config --derive` читает
фактический config, сохраняет отдельный датированный производный файл и отчёт
по изменённым полям. Исходный WWW config не редактируется. Производная меняет только
inspection budget ceilings, methods=anychange, sequential и необходимый timeout;
не придумывает source/checkpoint, python, модель, revision или image preprocessing.
Если несовместимы ресурсы/модель/native crop, статус blocked и точная причина в
config-report.json. Производный файл в таком случае не считается рабочим: исправьте
его копию по фактическим ресурсам и повторите check-config без --derive в новом каталоге.

```bash
(
  set -euo pipefail
  set -C
  cd /home/sasha/Facade_Visual_Inspection
  test ! -e runs/2026-11-16-cvpr-config-001
  : > runs/2026-11-16-cvpr-config-001.log
  /home/sasha/miniconda3/envs/change-agentic-rag/bin/python -u -s -B -m facade_change.agentic inspection check-config \
    --config /home/sasha/Facade_Change_Benchmark/configs/2026-11-08-agentic-wsl-16k-001.local.json \
    --protocol runs/2026-11-16-cvpr-cohort-001/full/protocol.json \
    --derive --out runs/2026-11-16-cvpr-config-001 \
    2>&1 | tee -a runs/2026-11-16-cvpr-config-001.log
)
```

Ни numeric validation, ни присутствие checkpoint-файла не подтверждают реальную
загрузку Qwen или VRAM fit. Preflight ниже проверяет полный public cohort и конфиг
без запуска моделей, включая native размеры для fixed_history. Smoke имеет отдельный
manifest/protocol и повторно проверяется внутри своего run.

```bash
(
  set -euo pipefail
  set -C
  cd /home/sasha/Facade_Visual_Inspection
  test ! -e runs/2026-11-16-cvpr-preflight-001
  : > runs/2026-11-16-cvpr-preflight-001.log
  /home/sasha/miniconda3/envs/change-agentic-rag/bin/python -u -s -B -m facade_change.agentic inspection preflight \
    --questions runs/2026-11-16-cvpr-cohort-001/full/export/questions.json \
    --histories /home/sasha/Facade_Change_Benchmark/data/2026-11-04-histories-s192-001 \
    --config runs/2026-11-16-cvpr-config-001/2026-11-16-inspection.local.json \
    --protocol runs/2026-11-16-cvpr-cohort-001/full/protocol.json \
    --out runs/2026-11-16-cvpr-preflight-001 \
    2>&1 | tee -a runs/2026-11-16-cvpr-preflight-001.log
)
```

Перед GPU smoke отдельно посмотрите процессы. Во время тяжёлого WWW-run smoke
не запускается; чужие процессы не останавливаются.

```bash
nvidia-smi
```

Ровно два случая и только heuristic/agent (зафиксированы smoke protocol):

```bash
(
  set -euo pipefail
  set -C
  cd /home/sasha/Facade_Visual_Inspection
  test ! -e runs/2026-11-16-cvpr-smoke-001
  : > runs/2026-11-16-cvpr-smoke-001.log
  /home/sasha/miniconda3/envs/change-agentic-rag/bin/python -u -s -B -m facade_change.agentic inspection run \
    --questions runs/2026-11-16-cvpr-cohort-001/smoke/export/questions.json \
    --histories /home/sasha/Facade_Change_Benchmark/data/2026-11-04-histories-s192-001 \
    --config runs/2026-11-16-cvpr-config-001/2026-11-16-inspection.local.json \
    --protocol runs/2026-11-16-cvpr-cohort-001/smoke/protocol.json \
    --out runs/2026-11-16-cvpr-smoke-001 \
    2>&1 | tee -a runs/2026-11-16-cvpr-smoke-001.log
)
```

Оценка smoke — отдельный CPU-шаг, включая completed_with_errors, все missing и отказы:

```bash
(
  set -euo pipefail
  set -C
  cd /home/sasha/Facade_Visual_Inspection
  test ! -e runs/2026-11-16-cvpr-smoke-eval-001
  : > runs/2026-11-16-cvpr-smoke-eval-001.log
  /home/sasha/miniconda3/envs/change-agentic-rag/bin/python -u -s -B -m facade_change.agentic inspection evaluate \
    --questions runs/2026-11-16-cvpr-cohort-001/smoke/export/questions.json \
    --answers runs/2026-11-16-cvpr-cohort-001/smoke/export/answers.private.json \
    --run runs/2026-11-16-cvpr-smoke-001 --out runs/2026-11-16-cvpr-smoke-eval-001 \
    2>&1 | tee -a runs/2026-11-16-cvpr-smoke-eval-001.log
)
```

Владелец сначала смотрит smoke report, reply/trace/costs, исходные RGB и native crop
geometry, реальные processor image_grid_thw, ошибки/OOM/retry и потребление памяти.
Полный run ниже **не входит в цепочку preflight/smoke** и запускается отдельным
решением после этой проверки и освобождения GPU от WWW. Это четыре режима:

```bash
(
  set -euo pipefail
  set -C
  cd /home/sasha/Facade_Visual_Inspection
  test ! -e runs/2026-11-16-cvpr-full-001
  : > runs/2026-11-16-cvpr-full-001.log
  /home/sasha/miniconda3/envs/change-agentic-rag/bin/python -u -s -B -m facade_change.agentic inspection run \
    --questions runs/2026-11-16-cvpr-cohort-001/full/export/questions.json \
    --histories /home/sasha/Facade_Change_Benchmark/data/2026-11-04-histories-s192-001 \
    --config runs/2026-11-16-cvpr-config-001/2026-11-16-inspection.local.json \
    --protocol runs/2026-11-16-cvpr-cohort-001/full/protocol.json \
    --out runs/2026-11-16-cvpr-full-001 \
    2>&1 | tee -a runs/2026-11-16-cvpr-full-001.log
)
```

Его оценка также отдельна:

```bash
(
  set -euo pipefail
  set -C
  cd /home/sasha/Facade_Visual_Inspection
  test ! -e runs/2026-11-16-cvpr-full-eval-001
  : > runs/2026-11-16-cvpr-full-eval-001.log
  /home/sasha/miniconda3/envs/change-agentic-rag/bin/python -u -s -B -m facade_change.agentic inspection evaluate \
    --questions runs/2026-11-16-cvpr-cohort-001/full/export/questions.json \
    --answers runs/2026-11-16-cvpr-cohort-001/full/export/answers.private.json \
    --run runs/2026-11-16-cvpr-full-001 --out runs/2026-11-16-cvpr-full-eval-001 \
    2>&1 | tee -a runs/2026-11-16-cvpr-full-eval-001.log
)
```

Сильное обобщение по 20–30 случаям и преимущество agent не установлены. Подготовка
smoke не завершает статью. Выбор и проверка 1–2 таблиц Вестника остаются отдельной работой.
