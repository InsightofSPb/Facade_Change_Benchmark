# Один approved-кейс с метриками, затем весь A → C → B — 2026-11-17

## Повтор после первого настоящего E2E

Настоящий `approved-acb-001` дошёл до offline score. На одном вопросе
`qa-3b640db4fe605600a6fbd098` (Большой проспект П.С., 83) A получил change
и structured_correct=true; B — no_change и false; C завершился технической
ошибкой `Use one action discriminator, type or action, not both`. Индекс
построен: 80 records, 149 vectors. Основной прогон не начат.

Присланы консоль и metrics.private.json, без исходных двух JSON-ответов C.
Нельзя утверждать, что значения type/action в них совпадали. Исправлена
самостоятельно воспроизводимая избыточная строгость обработчика: одинаковые
type/action и явная оболочка type=action принимаются без изменения выбранного
действия, query, reason или observation_id. Противоречащие значения отклоняются
с указанием обоих значений. Raw и нормализации сохраняются. Prompt, бюджеты,
reference и правила проверки содержимого не изменены. Transport revision:
`2026-11-17-consistent-action-discriminators-v5`.

После исправления прошли все 66 тестов `test_2026_11_15_www*.py`, включая
сохранение query/reason/IDs, реальный dispatch с scripted workers, отказ при
конфликте действий, reuse → smoke → score → continue и проверку неизменности
исходных файлов. Повторный настоящий Qwen здесь не выполнялся.

Повтор ниже сохраняет предыдущий запуск, побайтово переиспользует его public
и smoke-public экспорты и готовый index.json, сверяя их с исходными планами.
Новые планы фиксируют текущий код. Решения и источники остаются прежними;
review, export, embeddings повторно не выполняются. Повторяется тот же smoke,
выбранный до первого инференса, независимо от его результата.

```bash
cd /home/sasha/Facade_Change_Benchmark &&
git pull --ff-only &&
set -o pipefail &&
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -u scripts/2026-11-17-run-approved-www.py \
  --prepared-from /home/sasha/Facade_Change_Benchmark/runs/2026-11-17-www-approved-acb-001 \
  --smoke-only \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-17-www-approved-acb-002 \
  2>&1 | tee /home/sasha/Facade_Change_Benchmark/runs/2026-11-17-www-approved-acb-002.smoke.console.log
```

После технически успешного повторного smoke — без обновления кода между этапами:

```bash
cd /home/sasha/Facade_Change_Benchmark &&
set -o pipefail &&
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -u scripts/2026-11-17-run-approved-www.py \
  --continue \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-17-www-approved-acb-002 \
  2>&1 | tee /home/sasha/Facade_Change_Benchmark/runs/2026-11-17-www-approved-acb-002.main.console.log
```

На единственном smoke A засчитан только по структурному решению change, а не
по содержанию объяснения. Рамки A и B не выдали; region recall равен нулю.
Macro-F1 A=0.5 отражает фиксированные два класса при отсутствии примера no_change,
а не половину правильных ответов. Один кейс не даёт оценки общего качества.

## Предшествующее исправление nested final

В probe-006 Марата выполнил поиск и завершился supported; K61 после двух
поисков завершился insufficient_data. Локальный контроль дважды отклонён
за поле final. Полные ответы probe-006 в Cloud не передавались: из консоли
нельзя установить корректность содержимого этого поля.

Обработчик теперь поддерживает явную оболочку `final: {...}`, включая внешний
выбор answer. Вложенный ответ проходит прежнюю полную проверку: обязательные
поля, допустимые показанные источники, даты, координаты и поддержка. Неполный
объект не дополняется; конфликт retrieve/final отклоняется. Полный ответ может
завершить choose_action без ещё одного вызова модели. Raw и преобразования
сохраняются. Prompt revision не менялась; тогдашняя transport revision —
`2026-11-17-explicit-nested-final-v4`.

## Подтверждённые решения и фиксация протокола

`scripts/2026-11-17-run-approved-www.py` находит экспорт решений в Downloads
Windows/WSL или принимает явный `--decisions`. Разные найденные файлы требуют
явного пути. Проверяются исходные candidate bytes, число approved (по умолчанию
103), доступность ресурсов и пригодность видов в существующем RGB QA inventory.
Исходный экспорт копируется байт-в-байт. Candidate IDs, labels и wording не
переписываются; старые runs сохраняются.

`qa review --reviewed-gallery-only` меняет только объём HTML: страница содержит
просмотренные строки вместо повторного копирования изображений всех кандидатов.
`qa export --question-id` экспортирует подмножество уже approved состояний,
сохраняя исходную review-chain и reference; новых approvals не создаёт.

`www plan --include-reviewed` явно фиксирует весь approved-набор, допустимый
по RGB QA v2. Он допускает сохранённый cohort=unassigned и сохраняет исходные
QA exposure/evaluation eligibility. Неизвестная экспозиция не становится
none_found или audited evaluation. Техническая непригодность, pending identity
и несогласованные истории остаются причинами отказа. Отчёт агрегирует весь
зафиксированный набор под именем all_reviewed, сохраняя исходные статусы отдельно.
По переданному inventory все 103 выбранных вопроса eligible; исходные QA-статусы:
87 unknown и 16 known_qa_use. Изменений этих статусов нет.

## Порядок исполнения

`--condition-major --conditions photos_only bounded_agentic_retrieval fixed_retrieval`
означает весь A, затем весь C, затем весь B. Порядок входит в execution contract
и frozen plan. Подготовленная геометрия переиспользуется между проходами.
После прохода и при перехваченном прерывании сохраняется results.json с уже
завершёнными строками; отдельные результаты сохраняются и внутри прохода.
Несовместимый порядок не допускает resume. Если B ещё не выполнялся, его строки
остаются missing, а B−A/C−B отмечаются pending_condition; C−A можно посчитать.

Smoke выбирается до инференса: первый confirmed_change в порядке immutable
manifest, с предпочтением здания с verified архивным источником. Этот критерий
не использует прогнозы и не гарантирует полезность найденного источника.
Один кейс проходит экспорт → plan → A/C/B → offline score. Проверяется техника
полного цикла; одна строка не оценивает общее качество или статистическую значимость.
Нулевой quality score не блокирует продолжение; техническая ошибка блокирует.

## WSL

Первый запуск останавливается после одного кейса и печатает status, decision и
structured_correct для каждого режима. Метрики локализации и расходов сохранены
в smoke-score. Индекс V4.1 (80 записей, 77 assets) строится один раз; используется
тот же identity map, что в inventory. Логи всегда видимы.

```bash
cd /home/sasha/Facade_Change_Benchmark &&
git pull --ff-only &&
set -o pipefail &&
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -u scripts/2026-11-17-run-approved-www.py \
  --smoke-only \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-17-www-approved-acb-001 \
  2>&1 | tee /home/sasha/Facade_Change_Benchmark/runs/2026-11-17-www-approved-acb-001.smoke.console.log
```

После технически успешного smoke продолжение использует подготовленные входы
и индекс. Оно проверяет неизменность файлов и implementation hashes, затем
считает все 103 вопроса A → все C → все B и вызывает offline score даже при
отдельных ошибках инференса. Ошибки остаются в знаменателях.

```bash
cd /home/sasha/Facade_Change_Benchmark &&
set -o pipefail &&
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -u scripts/2026-11-17-run-approved-www.py \
  --continue \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-17-www-approved-acb-001 \
  2>&1 | tee /home/sasha/Facade_Change_Benchmark/runs/2026-11-17-www-approved-acb-001.main.console.log
```

`--prepare-only` ограничивает работу CPU-подготовкой review/export. Его каталог
не предназначен для --continue, которому нужен завершённый E2E smoke.

## Проверки и ограничения

62 теста `test_2026_11_15_www*.py` прошли после изменения порядка, включая
resume после остановки перед B и сохранение ровно исходной reference в smoke.
Дополнительный тест scoring прерванного A/C прошёл: B остаётся missing, его
сравнения не выводятся как измеренные. Всего 63 различных CPU-теста прошли.
CPU-цепочка с scripted workers проверила smoke-only,
offline score, continue без перестроения индекса, порядок AAAAA/CCCCC/BBBBB и
сохранение трёх намеренных ошибок среди 15 строк. Реальные модели не запускались.
Предшествующий CLI demo: 8 completed, 1 ожидаемый abstained, 0 errors.
Изменённые участки проверены `git diff --check`, команды — `bash -n`.

Настоящий GPU E2E и полный набор выполняются в WSL. Исправление формата не
подтверждает factual correctness, полезность архива или точность рамок.
