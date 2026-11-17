# WWW: реальные ответы, выбор действия и отдельный повтор C — 2026-11-17

Исправление транспорта и разделения этапов; не изменение вопросов, references,
состава выборки, моделей, бюджета или правил проверки свидетельств. GPU в Cloud
не запускался. Старый inference не переписывается и не возобновляется новым кодом.

## Что действительно находится в ZIP

`2026-11-17-www-agent-debug-001.zip`, SHA-256
`6b2ec6f4637d6564f55f8dce4feeec157a1f5a5df0b32747ae4bc32db45dbdc6`, содержит
499 файлов. Его public manifest содержит **103 вопроса: 85 temporal_change,
18 earliest_evidence**; earliest явно ограничены показанными локальными наблюдениями.
Private approvals/rejections и полный `run.json` в ZIP отсутствуют; сообщённые
103 approved / 3 rejected не пересоздавались и не менялись.

Фрагмент ZIP позднее приведённого в запросе снимка 36 started / 35 terminal:
здесь **54 результата C: 30 completed, 11 abstained, 13 error**, 120 Qwen calls.
Это не итог всех 103 и не полный A/C/B run. Все 54 `messages.json` согласованы
с точными requests/responses; SHA call-файлов сверены с `reply-*.json`.
После анализа все 499 файлов побайтово совпадают с ZIP.

Диагноз привязан к фактическому `contract.json`, а не к предположению по номеру
commit `6636d11`: хеши всех 11 implementation files совпали с `45a70b5` текущей
ветки до исправления. Recorded prompt — `2026-11-17-source-origin-action-v5`,
SHA `46b84a65c29c8674fc6dbaa7675fb73580175bf0c93b140c6a2dc5058244d96c`;
transport — `2026-11-17-consistent-action-discriminators-v5`.
Порядок A → C → B, `conditions_first`, Qwen3.5-4B, deterministic generation,
`enable_thinking=false`, 1024 output tokens, пять вызовов, два поиска и одна
попытка исправления ошибочного ответа сохранены.

На всех **70 запросах choose_action** поиск был разрешён: `retrieve_or_decide`,
подключённый архив, доступный бюджет. Однако локальная область всех вопросов
не требовала поиска. В saved normalized_action записаны **44 answer**, ни одного
retrieve. Ещё семь явных answer отклонены из-за null query; один неправильный
nested final ошибочно попал в ветку answer. Ledger retrieval_rounds, exposure.tools
и retrieval-файлы согласованно показывают **ноль инструментов**. Наличие query
не доказывает выбора retrieve; отсутствие поиска здесь не доказывает сбой dispatch.

## Конкретные технические причины

Во всех четырёх указанных случаях — `qa-e25ed6be7e10369e14262b83`,
`qa-66fd5c139802824a5c352ea7`, `qa-bf3eb6f3d929eed2d0a6410b`,
`qa-a24d588ef4b6b1003d83fd42` — первый raw одинаковой структуры:

```json
{"stage":"choose_action","action":"answer","query":null}
```

Парсер распознавал answer, затем выдавал
`choose_action: missing fields=[]; unexpected fields=['query']`.
Такой же отказ встретился ещё в трёх случаях. После штатного retry четыре
перечисленных случая дали соответственно completed / abstained / completed /
abstained. Поэтому исправление семи **вызовов** нельзя выдавать за исправление
семи итоговых errors.

Теперь только `query:null` у явно выбранного answer/abstain считается пустым
неиспользуемым аргументом. Raw/parsed сохраняются; отдельная нормализация
`removed_null_query_from_answer` документирует транспортное изменение.
Непустая строка, пустая строка, массив, boolean или другой non-null query у
answer/abstain отклоняются с конкретным пояснением. Действие не меняется,
retrieve автоматически не подставляется. Неизвестные аргументы и противоречащие
друг другу discriminators продолжают отклоняться.

В одном сохранённом ответе было `{"stage":"answer","final":{"type":"answer",...}}`.
Внутри лежали произвольные change/no_change/boxes вместо публичного final.
Старый normalize_final извлекал объект, затем dispatcher ошибочно считал его
выбором answer. Теперь явный final с другим type отклоняется; рамки/даты из него
не удаляются и не подставляются в следующий ответ. Это ужесточение границы этапов.

До исправления choose_action получал весь длинный SYSTEM, включая final dates,
uncertainty, citations и bbox. Одновременно последний control просил только
действие. Эта конкуренция инструкций реальна; её причинное влияние на конкретные
генерации ещё требует WSL smoke. В C теперь system выбирается по этапу:

- choose_action: те же фактические требования и origin/scope rules, затем только
  retrieve/answer/abstain. Полная схема final и bbox-инструкции здесь отсутствуют;
- answer: прежние требования к содержимому и строгой геометрии, без инструкции
  выбора действия. Последний control явно требует значения плоского final,
  массивы citations/observation_ids/boxes/limitations и корректное JSON-экранирование.

A/B продолжают получать прежний SYSTEM побайтово. Общие фактические требования
берутся из него же; локальные фото, вопрос и task_scope не меняются. Хеши обоих
system C записаны в `agentic_stage_prompts` с revision
`2026-11-17-stage-scoped-contract-v6`, transport — `2026-11-17-null-unused-query-v6`.
`reply-*.json` дополнительно хранит allowed_action и generation diagnostics.

Из 13 **сохранённых итоговых errors**: четыре — невалидный JSON, три — отсутствует
date_interval, два — citations не список; по одному — лишний uncertain, limitations
не список, неверная дата и недостаточная общая поддержка рамки. Эти содержательные
и структурные ошибки не исправляются удалением полей или реконструкцией ответа.
Восемь malformed JSON replies завершились на 372–464 tokens из 1024;
`reached_max_new_tokens=false` у всех. В частности, nested JSON не закрыт до конца.
Гипотеза об усечении по token limit этими данными не подтверждается. Лимит не повышен,
закрывающие скобки не дописываются; другой stop reason из этих полей не выводится.

## Replay и неподтверждённые архивные утверждения

`scripts/2026-11-17-www-action-replay.py` сверяет public SHA, requests/messages,
сохранённые parse traces, evidence именно показанного вызова, tools и costs.
`--baseline-ref` загружает только локальный исходник указанного commit и требует
совпадения его SHA с contract; workers, retrieval, alignment не вызываются.

| Проверка 120 сохранённых ответов | Baseline | Исправленный код |
|---|---:|---:|
| Принятый выбор действия | 45 | 51 |
| Public fields прошли на записанной coverage statistics | 26 | 26 |
| Для завершения проверки рамки нужны отсутствующие support pixels | 19 | 19 |
| Отклонённый ответ | 30 | 24 |

Разница — ровно семь null-query choices, теперь принятых, и один неправильный
nested final, теперь отклонённый. Все 26 результатов проверки public fields
побайтово эквивалентны по JSON значениям. Первая колонка включает старый ошибочный
fallback nested final → answer; это не корректный final и не поиск.

В ZIP нет RGB, preparation/common_support и per-photo support arrays. Для .any()
используется **записанный** common_pixels из исходного запроса, не выдуманная маска.
Проверка footprint останавливается с `support_validation_unavailable`. Среди этих
19 ответов четыре имели сохранённый отказ по common support; он остаётся виден,
но без пикселей повторно не подтверждается. Ни один такой ответ не объявлен valid.
С доступными сохранёнными masks скрипт проверяет SHA/grid и вызывает тот же строгий
валидатор. Source/date/citation/owner/shape/bounds правила не ослаблены.

Шесть final содержат формулировки об отсутствии архивных снимков при not_searched.
Например: «Отсутствие архивных снимков внутри этого интервала не позволяет исключить
скрытые изменения, но подтверждает стабильность видимого состояния на момент съёмки».
Отсутствие не установлено поиском, а причинная связка со стабильностью не обоснована.
Это отдельная factual-support проблема, не runtime error. В replay такие фразы
вынесены в `archive_absence_claims_for_human_review`; эвристика не меняет decision,
статус или человеческие labels. Упоминание недостатка **предоставленных** источников
и утверждение об отсутствии источников во всём архиве нужно различать при review.

Результат Cloud:
`/workspace/work/2026-11-17-www-action-replay-cpu-004/{cases.private.csv,replay.private.json,summary.json}`.
CSV содержит каждую из 54 строк: допуск поиска, raw discriminators, принятые действия,
отказы парсера, tool counts и final status. JSON содержит все raw replies и оба replay.
Это технический аудит наблюдённой траектории, не новый прогон и не предсказание
того, что модель ответит после устранения retry.

## Выполненные CPU-проверки

```bash
/workspace/work/facade-venv/bin/python -u -s -B -m unittest tests.test_2026_11_17_www_action_debug tests.test_2026_11_15_www tests.test_2026_11_15_www_curation -v
/workspace/work/facade-venv/bin/python -u -s -B -m unittest tests.test_2026_11_15_www_smoke -v
/workspace/work/facade-venv/bin/python -u -s -B -m facade_change.agentic www demo --out /workspace/work/2026-11-17-www-agent-debug-cpu-demo-002
/workspace/work/facade-venv/bin/python -u -s -B scripts/2026-11-17-www-action-replay.py --run /workspace/work/2026-11-17-www-agent-debug-001/inference --baseline-ref 45a70b5 --out /workspace/work/2026-11-17-www-action-replay-cpu-004
/workspace/work/facade-venv/bin/python -u -s -B scripts/2026-11-17-www-action-replay.py --run /workspace/work/2026-11-17-www-agent-debug-cpu-demo-002/inference --out /workspace/work/2026-11-17-www-action-replay-synthetic-002
```

51 + 10 = **61 CPU tests OK**. Включены настоящий dispatch с scripted pool,
локальный answer без поиска, лимиты и retry, строгие даты/цитаты/геометрия,
C-only plan → run → score, отказ старого A/B/C plan для нового C и сохранение
байтов вопросов/references. CLI demo: 8 completed, 1 ожидаемый abstained в A,
0 errors, `models_executed=false`. Replay demo проверяет также полные сохранённые
support masks: 15 replies, 6 принятых действий, 9 valid finals. Дополнительно
сравнены model-facing messages шести A/B вызовов на одинаковом CPU fixture
в baseline `45a70b5` и новом коде: полное совпадение. Это не проверка поведения
настоящего Qwen. `git diff --check`, синтаксис Bash-команд и встроенного Python
также проверены.

## WSL: сначала три диагностических вопроса

Команды выполнять после завершения или сохранённой остановки текущего процесса:
не обновлять его исходники во время исполнения. Старые каталоги остаются на месте.
Используются **существующие** два архивных вопроса и один локальный контроль из
tool-probe-inputs-001; 103 approved questions не редактируются и не подменяются ими.

```bash
cd /home/sasha/Facade_Change_Benchmark &&
git pull --ff-only &&
set -o pipefail &&
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -u -m facade_change.agentic www run \
  --questions /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-tool-probe-inputs-001/questions.json \
  --config /home/sasha/Facade_Change_Benchmark/configs/2026-11-15-www-archive-20mp-001.local.json \
  --archive-index /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-archive-index-v31-001/index.json \
  --conditions bounded_agentic_retrieval \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-17-www-agent-debug-smoke-001 \
  2>&1 | tee /home/sasha/Facade_Change_Benchmark/runs/2026-11-17-www-agent-debug-smoke-001.console.log
```

Смотреть calls, reply, retrieval, exposure и final: модель действительно выбрала
retrieve → dispatcher вызвал search → появились archive IDs → final прошёл
прежние проверки. Поиск программно не вставляется. Корректный мотивированный отказ
допустим, но если retrieve не выбран, этот smoke не подтверждает реальный retrieval
путь. Для локального контроля answer без поиска допустим. Ноль runtime errors
не заменяет проверку смысла ответа, особенно archive absence при not_searched.

## Отдельный новый C и его оценка

После просмотра smoke — новая execution revision на **тех же bytes** 103 вопросов
и references. Индекс V4.1 переиспользуется, embeddings не перестраиваются. Ни
qa review/export, ни изменения approved/rejected не выполняются. Подготовка
разрешает C-only frozen plan; score обрабатывает только его объявленные условия.
Все 103 строки C, включая error/missing, остаются в знаменателе. Отсутствующие в
этом плане A/B помечаются `not_planned_in_this_execution`, не рисуются нулевыми
метриками и не подмешиваются из другого run. Остальные проверки plan/contract,
времени freeze, данных, identity и references сохраняются.

Команда ниже предварительно сравнивает вопросы, ответы, config, index и identity
с исходными plan/contract, затем создаёт **новый** plan до любого inference.
Она разрешает score после exit code 1 (сохранённые case errors), сохраняя их в
результате; другие ошибки не игнорирует. stdout/stderr видимы в foreground.

```bash
cd /home/sasha/Facade_Change_Benchmark &&
set -o pipefail &&
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -u - <<'PY' 2>&1 | tee /home/sasha/Facade_Change_Benchmark/runs/2026-11-17-www-c-stage-v6-001.console.log
import json
from pathlib import Path
import shlex
import subprocess
import sys
from facade_change.agentic.config import load_config
from facade_change.io import sha256

root = Path('/home/sasha/Facade_Change_Benchmark')
old = root / 'runs/2026-11-17-www-approved-acb-002'
out = root / 'runs/2026-11-17-www-c-stage-v6-001'
contract = json.loads((old / 'inference/contract.json').read_text())
plan = json.loads((old / 'plan/plan.private.json').read_text())
questions, answers = old / 'public/questions.json', old / 'public/answers.private.json'
index = old / 'archive-index/index.json'
config = root / 'configs/2026-11-15-www-archive-20mp-001.local.json'
inventory = root / 'runs/2026-11-16-www-home-v41-001/review/inventory/inventory.json'
identity = Path(contract['identity_source']['path'])
assert not out.exists(), 'Use a new output directory'
assert contract == plan['execution_contract']
assert sha256(questions) == contract['questions_sha256'] == plan['questions_sha256']
assert sha256(answers) == plan['answers_sha256']
assert sha256(inventory) == plan['inventory_sha256']
assert sha256(index) == contract['archive_index_sha256']
assert sha256(identity) == contract['identity_source']['sha256']
assert load_config(config) == contract['config']
assert len(json.loads(questions.read_text())['questions']) == 103
def command(*args, allow_case_errors=False):
    argv = [sys.executable, '-u', '-m', 'facade_change.agentic', 'www', *map(str, args)]
    print(shlex.join(argv), flush=True)
    code = subprocess.run(argv, cwd=root).returncode
    if code not in ((0, 1) if allow_case_errors else (0,)):
        raise SystemExit(code)
    return code
shared = ['--questions', questions, '--config', config, '--archive-index', index,
          '--identity-map', identity, '--conditions', 'bounded_agentic_retrieval', '--condition-major']
command('plan', *shared, '--answers', answers, '--inventory', inventory,
        '--include-reviewed', '--out', out / 'plan')
status = command('run', *shared, '--out', out / 'inference', allow_case_errors=True)
command('score', '--questions', questions, '--answers', answers, '--run', out / 'inference',
        '--plan', out / 'plan/plan.private.json', '--out', out / 'score')
raise SystemExit(status)
PY
```

A/B повторять сейчас не требуется: их system text, control, модель, входы и
строгий final validator не изменены. Их сохранённые результаты оцениваются
со **своим** исходным plan/contract после terminal состояния старого run.
У C изменён фактор «stage-specific instructions + action transport»; это новая
версия метода. Старые A/B и новый C не являются одним неизменным execution.
Нельзя копировать results, подменять contract, использовать старый plan для C
или `--resume-from` старого запуска. Автоматических межверсионных paired metrics
здесь нет: для такого сравнения нужно отдельно зафиксировать совместимость
cohort/reference/identity/model/budgets и назвать изменённый фактор. Необходимость
повтора A/B возникает при изменении их входов/модели/промпта/валидатора или
выявленной несовместимости; таких оснований в этом исправлении не установлено.
