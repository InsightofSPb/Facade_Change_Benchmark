# Анализ пары, история областей и ограниченный агент — 2026-11-08

Рабочая ветка: `codex/2026-11-08-change-agentic-rag`, исходная ветка
`codex/2026-10-24-main-workflow`, исходный commit
`b2d3acebfc27eabba3955f2366ee0c8e2c9e1689`.

## Что запускается

`python -m facade_change.agentic` принимает новую пару RGB без GT либо ID изображений
подготовленного manifest. Сначала выполняются существующие alignment/MAGSAC и методы
через `RemotePool`, затем освобождаются их workers и выполняются режимы Qwen.
Верхнеуровневый workflow также принимает `benchmark agentic`, `prepare agentic-index`
и `gallery agentic`. Нового планировщика экспериментов нет: используются имеющиеся
RunStore, журналы, геометрия, методы и генератор HTML-галерей.

| Модуль | Ответственность |
| --- | --- |
| `agentic/pipeline.py` | raw pair, manifest, готовые canonical histories, карты отдельных методов и применение замороженного fusion |
| `agentic/store.py` | постоянная область, датированные наблюдения, отдельные типы свидетельств, журнал с UUID хранилища |
| `agentic/retrieval.py` | точная история, cosine-поиск, лексический поиск, последовательности признаков, понятия и источники |
| `agentic/models.py`, `model_worker.py` | реальные Qwen, SAM3, Jina, существующий DinoEncoder; отдельные интерпретаторы и владение GPU |
| `agentic/tools.py`, `reasoning.py` | разрешённые инструменты, бюджеты, проверка evidence IDs, сохранение и восстановление CPU-диалога |
| `agentic/report.py`, `evaluation.py` | просмотр исходных артефактов, сравнение фиксированного набора случаев, отдельный доступ к GT |

Минимальная конфигурация — `configs/2026-11-08-agentic.example.json`. В ней SIFT/CPU,
SSIM и LZMA, три режима Qwen и последовательная передача GPU. Порог SSIM 0.15 —
**диагностический**, не подобранный baseline исследования. Без явного `fusion.run`
отчёт не выдаёт его за действующий fusion. AnyChange сохраняет SAM первого поколения.
Семантика, SAM3, бинарная карта AnyChange, перцептивные оценки и компрессионные величины
хранятся отдельно с исходными типами и происхождением.

Для fusion принимается завершённый `canonical_history_fusion` и выбранный variant,
например `stack--previous`. Используется текущая реализация `predict` из
`data_analysis/2026-11-06_history_fusion.py`, проверяются контракт методов, порядок
признаков, файлы модели, TRAIN fit и VAL threshold. Рецепт, которому нужны недоступные
исторические признаки, явно отклоняется. Новый fit и настройка на TEST не выполняются.

## Данные и границы вывода

ID области определяется зданием, видом, anchor RGB и фиксированным bbox. Для повторного
raw наблюдения нужен тот же anchor или сохранённое явное соответствие. Похожие
эмбеддинги не устанавливают идентичность участка. Готовые canonical histories
импортируются с их ID, преобразованиями и кропами без пересоздания RGB.

Наблюдение содержит `date`, `date_status`, `available_at`, RGB/support hashes и обе
стороны преобразования. Геометрическая поддержка не означает видимость за деревом,
отсутствие дефекта или сопоставимость освещения. Новая поддержка не становится
положительной меткой изменения. Даты, известные только до года, трактуются
консервативно как конец этого года при сравнении с точной датой.

История исключает текущую пару, будущие и недатированные сведения, проверяет доступность
цепочки свидетельств. Прогноз модели, измерение, `agent_hypothesis` и
`human_confirmation` — разные типы. Подтверждение требует автора и источника.
Импортированные записи неизменяемы: для другой политики доступности используйте новое
хранилище. Локальный store не поддерживает несколько параллельных писателей.

Аналоги берутся из TRAIN с отсечением по времени, исключением целевого здания и хешей
текущих изображений. DINO и Jina имеют разные versioned spaces. Исторический baseline —
детерминированная последовательность нормированных признаков, разностей, логарифмов
временных интервалов и индикаторов заполнения; это не обученная модель деградации.
Без эмбеддингов точная история и лексический поиск работают; векторный поиск возвращает
явную нехватку представления. Пустой корпус не заполняется фиктивными векторами.

База знаний первоначально содержит только `configs/semantic_class_groups.json`,
помеченный как проектная таксономия с датой доступности 2026-10-11. Для более раннего
cutoff этот документ недоступен. Учебники и подтверждённые причинные утверждения
не предоставлены. Их фрагменты можно импортировать отдельно:

```json
{
  "document_id": "unique-source-revision",
  "available_at": "2026-11-08",
  "source": {"kind": "user_document", "title": "Название реального документа", "locator": "Его действительный путь или URL"},
  "concepts": [{"concept_id": "local-concept", "name": "Термин из источника", "synonyms": []}],
  "relations": [],
  "fragments": [{"fragment_id": "fragment-1", "text": "Заменить дословным проверенным фрагментом", "concept_ids": ["local-concept"], "locator": "Страница или раздел"}]
}
```

Это схема, не готовый материал для импорта. `source.kind` допускает `project_taxonomy`,
`textbook`, `publication`, `user_document`; отношения имеют `subject`, `predicate`,
`object`. Команда `knowledge-import --fragments /абсолютный/путь.json --out ...`
проверяет источники, ссылки понятий и сохраняет происхождение.

## Режимы и ресурсы

* `pair`: два изображения, даты и минимальные метаданные; без истории и инструментов.
* `fixed_context`: та же модель, детерминированно добавленные история, карты и знания.
  Все сводки методов сохраняются; не более двух preview-карт попадают в prompt.
* `agent`: та же исходная пара; модель выбирает разрешённые инструменты истории,
  открытия RGB, карт, поиска, знаний и текстового SAM3.

Изображения передаются через настоящий `AutoProcessor`, не только текстом пути.
Лимиты по умолчанию: 4 действия, 5 логических модельных запросов (включая embeddings
и SAM3), 8 изображений и 4096 входных токенов на prompt, 1024 выходных токена на ответ,
32 изображения / 16384 входных / 4096 выходных токенов суммарно, 300 секунд.
`model_requests` считает запросы, а `model_calls` — фактические вызовы, включая OOM retry.
Переполнение даёт явное завершение с недостатком данных; precision и разрешение
не уменьшаются автоматически. При изменении бюджета согласуйте лимиты `models.qwen`
с `budgets` в новой конфигурации.

Qwen закреплён на `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, BF16/SDPA,
`enable_thinking=False`, greedy decoding, batch 1. Размер изображения для процессора
задан явно: 65536–262144 пикселя. SAM3 закреплён на
`660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`: builder `device="cuda"`, FP32 параметры,
BF16 autocast, resolution 1008, без интерактивности и компиляции. Проверяются
установленные исходники и checkpoint; поле `masks_logits` этой версии трактуется
как результат sigmoid. Исходные маски и SAM3 сохраняются отдельно, включая
обратное преобразование в координаты наблюдения и anchor.

В `sequential` смена модели завершает принадлежащий приложению worker. CPU-диалог,
артефакты и evidence остаются на диске, Qwen восстанавливает входы без переноса KV-cache.
В `resident` одновременно допускаются только Qwen и SAM3; вызовы последовательны.
Для одного идемпотентного OOM-запроса разрешён ровно один повтор после освобождения
собственных workers. Чужие процессы не завершаются. Между режимами workers закрываются;
переиспользование дискового кеша явно учитывается и не называется холодным запуском.

## Получение кода и конфигурация в WSL

Команды ниже выполняются в Bash. Использованы предоставленные владельцем абсолютные
пути. Реальные изображения, готовые histories и fusion в Cloud отсутствуют; их пути
нужно задать явно. Не запускайте эти команды в неисправной среде `main`.

```bash
cd /home/sasha/Facade_Change_Benchmark
git fetch origin
git switch codex/2026-11-08-change-agentic-rag
git pull --ff-only
PY=/home/sasha/miniconda3/envs/change-agentic-rag/bin/python
CFG=/home/sasha/Facade_Change_Benchmark/configs/2026-11-08-agentic.local.json
RUNS=/home/sasha/Facade_Change_Benchmark/runs
"$PY" -m facade_change.agentic configure --out "$CFG" --gpu-mode sequential
```

`configure` отказывается перезаписывать файл. Для повторного опыта выберите новое имя
конфига/каталога. Родительскому CPU-процессу нужны зависимости `requirements-cpu.txt`;
Qwen/SAM3 используют свои заданные интерпретаторы. Если зависимости CPU отсутствуют
в `change-agentic-rag`, используйте существующий рабочий интерпретатор с этими
зависимостями как `PY`; пути workers в конфигурации менять не требуется.

Чтобы включить готовые детекторы вместо минимальной диагностики, создайте другой конфиг.
Путь методов взят из документа подготовки 2026-11-04; существование файла проверяется
на вашей машине. Он не создаётся автоматически из предположений:

```bash
: "${DETECTOR_PY:?Укажите абсолютный путь проверенного Python существующих детекторов}"
"$PY" -m facade_change.agentic configure \
  --out /home/sasha/Facade_Change_Benchmark/configs/2026-11-08-agentic-methods.local.json \
  --methods-config /home/sasha/Facade_Change_Benchmark/configs/2026-11-04-models.local.json \
  --methods ssim lzma_abs anychange semantic_dino dinov2 lpips \
  --alignment-python "$DETECTOR_PY" --detector-python "$DETECTOR_PY"
```

`--detector-python` явно переопределяет Python всех выбранных методов; если существующий
конфиг уже разделяет нужные среды, опустите этот флаг и задайте только alignment Python.
Для готового semantic run предусмотрен `--semantic-run`; для готового fusion —
`--fusion-run /действительный/каталог --fusion-variant stack--previous`. Набор и порядок
методов должны совпадать с сохранённым fusion. MSDZip подключается своим существующим
именем и параметрами из methods config; случайное среднее карт не применяется.

## Восемь локальных проверок

1. Конфигурация, импорты, локальные веса. Первый вызов проверяет зависимости контроллера;
   `check` проверяет реальные окружения workers и локальный кеш, ничего не скачивает.
   `--load-model` дополнительно загружает модели последовательно, но не проверяет качество.

```bash
"$PY" -c 'import numpy, PIL, cv2, scipy, pycocotools; print("CPU imports OK")'
"$PY" -m facade_change.agentic check --config "$CFG" \
  --roles qwen sam3 --out "$RUNS/2026-11-08-agentic-check-001"
```

2. Новая пара без GT. Задайте `REFERENCE`, `SOURCE`, `REFERENCE_DATE`, `SOURCE_DATE`,
   `BUILDING_ID`, `VIEW_ID` действительными значениями. Даты должны быть подтверждены;
   более ранний кадр — reference. Это параметризованная команда, а не выдуманные пути.

```bash
: "${REFERENCE:?Абсолютный путь раннего RGB}" "${SOURCE:?Абсолютный путь позднего RGB}"
: "${REFERENCE_DATE:?Подтверждённый YYYY или YYYY-MM-DD}" "${SOURCE_DATE:?Подтверждённая дата}"
: "${BUILDING_ID:?ID физического здания}" "${VIEW_ID:?ID вида}"
"$PY" -m facade_change.agentic run --config "$CFG" \
  --reference "$REFERENCE" --source "$SOURCE" \
  --reference-date "$REFERENCE_DATE" --source-date "$SOURCE_DATE" \
  --building-id "$BUILDING_ID" --view-id "$VIEW_ID" --modes pair \
  --out "$RUNS/2026-11-08-agentic-pair-001"
```

Опционально задайте `--split train|val|test`, только если принадлежность проверена,
`--bbox x0 y0 x1 y1` в native anchor, `--region-id` и `--alignment-run` для готовой
геометрии. По умолчанию split `unassigned`, а bbox охватывает reference. Изображение
слишком большой области вызывает явный отказ: для тонких дефектов выберите явный bbox.
Из manifest используйте `--manifest /абсолютный/manifest.json --reference-id ID --source-id ID`.
Входные поля проецируются по разрешённому списку; GT в raw cases отклоняется до инференса.

3. Реальная история минимум с тремя датами из существующего `history_features`:

```bash
: "${HISTORIES:?Абсолютный каталог готовых canonical histories}"
: "${TRACK_ID:?Существующий track_id из его manifest.json}"
"$PY" -m facade_change.agentic history --config "$CFG" \
  --histories "$HISTORIES" --track-id "$TRACK_ID" --min-dates 3 \
  --available-at observation_date --out "$RUNS/2026-11-08-agentic-history-001"
```

`observation_date` — явное ретроспективное допущение доступности снимка в дату съёмки.
Вместо него можно задать реальную ISO-дату доступности архива; прошлые запросы тогда
не получают этот архив раньше срока. Для уже сохранённых DINO-признаков добавьте
`--features /абсолютный/каталог/features`. Команда не использует targets архива и не
выбирает track по GT. Пользовательский выбор существующего track не доказывает полноту
по всему фасаду; исходный способ подготовки/выбора архива остаётся частью provenance.

4. Все три режима на одной зафиксированной паре:

```bash
"$PY" -m facade_change.agentic run --config "$CFG" \
  --reference "$REFERENCE" --source "$SOURCE" \
  --reference-date "$REFERENCE_DATE" --source-date "$SOURCE_DATE" \
  --building-id "$BUILDING_ID" --view-id "$VIEW_ID" \
  --modes pair fixed_context agent --out "$RUNS/2026-11-08-agentic-all-001"
```

Вместо реальной пары доступен `smoke --kind models` с тремя процедурными изображениями.
Он вызывает настоящие configured models; заранее подготовленных ответов в CLI нет.

5. Принудительная последовательность настоящих вызовов Qwen → SAM3 → Qwen:

```bash
"$PY" -m facade_change.agentic smoke --config "$CFG" --kind handoff \
  --out "$RUNS/2026-11-08-agentic-handoff-001"
```

Синтетический цветной участок проверяет передачу ресурсов, не обнаружение реальных
повреждений. Для проверки совместного удержания создайте новый конфиг с
`configure --gpu-mode resident` и новым `--out`, затем повторите эту команду с ним.

6. Освобождение Qwen, SAM3 и восстановление Qwen из CPU-сообщений:

```bash
"$PY" -m facade_change.agentic smoke --config "$CFG" --kind recovery \
  --out "$RUNS/2026-11-08-agentic-recovery-001"
```

Этот вариант принудительно использует sequential и перечитывает сохранённый transcript.
Проверьте событие `cpu_transcript_restored`, `worker_logs/pool.jsonl`, загрузки и PID:
вторая генерация должна принадлежать новому Qwen worker. OOM искусственно не вызывается;
ограниченный retry проверяется CPU-тестом и фиксируется отдельно при реальном OOM.

7. Максимальный настроенный входной контекст:

```bash
"$PY" -m facade_change.agentic smoke --config "$CFG" --kind context \
  --out "$RUNS/2026-11-08-agentic-context-001"
```

Заполняются `max_images` и текст до `max_input_tokens` с измерением настоящим процессором.
Все пробы, фактическое число токенов и память сохраняются. При генерации запрашивается
`max_output_tokens`, но ранний EOS подтверждает только реально полученную длину.
Повторённые синтетические изображения и текст проверяют вместимость; разнообразный
длинный агентный диалог требует отдельного прогона. Прежний тест с 0.736 GiB свободной
памяти не считается проверкой этих лимитов.

8. Сравнение baseline и трёх режимов, затем отдельный отчёт:

```bash
"$PY" -m facade_change.agentic compare \
  --runs "$RUNS/2026-11-08-agentic-all-001" \
  --cases "$RUNS/2026-11-08-agentic-all-001/input_manifest.json" \
  --out "$RUNS/2026-11-08-agentic-comparison-001"
"$PY" -m facade_change.agentic report \
  --run "$RUNS/2026-11-08-agentic-all-001" \
  --out "$RUNS/2026-11-08-agentic-report-001"
```

`--labels /абсолютный/labels.json` добавляется только к `compare`. Без него precision,
recall и IoU неизвестны; таблицы состояния и затрат доступны. Схема labels:
`{"label_scope":"region_temporal_change","provenance":{...},"cases":[{"case_id":"...","change":true}]}`.
Неизвестное изменение — `null`. Для пиксельного сравнения нужны реальные `mask_path`,
`support_path`, `mask_task:"temporal_change"` и совпадающая `coordinate_system`.
Текст Qwen и семантическая маска SAM3 не превращаются автоматически в temporal mask.

Когорта берётся из входного manifest, независимо от успешности подготовки. Неподготовленные
случаи и ошибки остаются в знаменателях. Сохраняются facade-macro metrics, coverage,
abstention, условные метрики только решённых случаев, evidence/tool metrics и затраты.
Полнота относится к перечисленной когорте: для оценки предложения кандидатов в неё
нужно заранее включить все целевые области, включая не предложенные системой.

## Необязательный Jina и существующий DINO

Jina использует реальные `encode_image` / `encode_text(task="retrieval.query")`,
локальные веса и закреплённый удалённый код. Нужны `torch`, `transformers`,
`huggingface_hub`, `einops`, `timm`, Pillow и зависимости закреплённой реализации.
Flash-attention/xformers по умолчанию отключены. Адаптер не устанавливает пакеты.
Кроме весов `jinaai/jina-clip-v2` нужны локальные code/config snapshots:

| Ресурс | Revision |
| --- | --- |
| `jinaai/jina-clip-v2` | `e10d47f5691d0454a0fb5d13f46f2199b74cb436` |
| `jinaai/jina-clip-implementation` | `39e6a55ae971b59bea6e44675d237c99762e7ee2` |
| `jinaai/jina-embeddings-v3` — config, не отдельные веса | `ab036b023d30b4d1138c4c3bfa9f0c445ab455d6` |
| `jinaai/xlm-roberta-flash-implementation` | `bd55a5ec8e6c0fb1d6c26efb4b6a4a74ce8a88d3` |

В Cloud этих ресурсов нет. Когда они подготовлены локально:

Для аналогов сначала нужны разрешённые TRAIN-наблюдения других зданий в хранилище.
Индекс, содержащий только целевую пару, корректно даст пустой результат после исключений.

```bash
"$PY" -m facade_change.agentic check --config "$CFG" --roles jina --load-model \
  --out "$RUNS/2026-11-08-agentic-jina-check-001"
: "${CUTOFF:?Дата доступности корпуса YYYY-MM-DD}"
"$PY" -m facade_change.agentic index --config "$CFG" --encoder jina \
  --cutoff "$CUTOFF" --out "$RUNS/2026-11-08-agentic-jina-index-001"
"$PY" -m facade_change.agentic search --config "$CFG" --encoder jina \
  --query 'crack in plaster' --cutoff "$CUTOFF" --exclude-building "$BUILDING_ID" \
  --out "$RUNS/2026-11-08-agentic-jina-query-001"
```

Для автоматического получения query-вектора агентом задайте
`configure --retrieval-encoder jina`. Визуальный DINO использует существующий
`history_features.DinoEncoder`; `--retrieval-encoder dino` и `index --encoder dino`
требуют корректной секции `models.dino` из существующего methods config. Импорт готовых
DINO features не требует вычисления новых векторов. Текстовый поиск без Jina:
`search --config "$CFG" --query 'crack' --cutoff "$CUTOFF"`.

## Артефакты и проверка в Cloud

Обычный запуск сохраняет `run.json` с конечным статусом, `config.json`, source snapshot,
`input_manifest.json`, `summary.json`, `cases/<case_id>/case.json`, карты и support.
Для каждого режима сохраняются `messages.json`, `state.json`, `trace.jsonl`,
`evidence.json`, `result.json`, `costs.json`; HTML — `report/report.html`.
Workers сохраняют журналы и provenance моделей. SAM3 пишет исходные вероятности,
маски, координатные преобразования и визуализации. Comparison создаёт `comparison.json`,
`comparison.csv`, `facades.csv`, `cases.csv`, `classes.csv`, `report.html`, `run.json`.
Smoke хранит `fixture/`, `verification.json` и `analysis/` либо `acceptance/` с журналом.

В затратах различаются общая подготовка, её геометрическая часть и дополнительная
стоимость режима. `pair` не приписывает себе методы, которые ему не передавались;
fixed/agent имеют также явную стоимость с общей подготовкой. Восстановление,
retrieval, SAM3 и retry входят в расходы режима. Непомеренная память — `null`.
Кеш-попадание имеет текущую нулевую модельную стоимость, исходные расходы сохраняются
отдельно. Сравнение кешированного запуска не является сравнением холодного старта.

Доступный CPU-пример без моделей:

```bash
"$PY" -m facade_change.agentic smoke --kind cpu \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-08-agentic-cpu-001
"$PY" -s -B -m unittest discover -s tests -v
```

CPU smoke действительно выполняет SIFT, SSIM, LZMA на трёх процедурных RGB и двух
парах, сохраняет одну постоянную область, историю и отчёт. Тесты проверяют GT-изоляцию,
cutoff, future-prefix invariance, стабильность ID, импорт настоящего формата histories,
маски/координаты, бюджеты, восстановление, OOM retry, инвалидацию кеша и учёт ошибок.
Тестовые подмены моделей существуют только в тестах. Настоящие Qwen, SAM3, DINO,
Jina, GPU-детекторы и fusion на реальных данных должны пройти локальную приёмку;
результаты CPU не доказывают их качество и совместимость CUDA.

### Выполнено в Cloud

Проверен чистый commit `1fd6f603b44548dc2f22a97074dd08a3d1b89573`
в CPU-среде Python 3.12, без PyTorch и весов моделей:

| Проверка | Результат |
| --- | --- |
| `python -s -B -m unittest discover -s tests -v` | 1238 тестов за 358.738 секунды; `OK (skipped=120)` |
| `python -m facade_change.agentic smoke --kind cpu --out <новый каталог>` | Две пары за 4.616 секунды, без ошибок; одна область, три наблюдения; SIFT/SSIM/LZMA и HTML |
| `python -m facade_change.agentic compare --runs <analysis> --cases <analysis/input_manifest.json> --out <новый каталог>` | Сравнение и HTML созданы; 8 строк для двух случаев × четырёх методов, отсутствие VLM сохранено явно |
| `python -m facade_change workflow benchmark agentic --help`, `compileall`, `git diff --check` | Успешно |

Пропуски относятся к недоступным необязательным зависимостям; это не успешные проверки
настоящих моделей. Smoke записал указанный commit и `dirty: false`. Успешное сравнение
без labels подтверждает обработку артефактов, не accuracy/IoU. Qwen, SAM3, Jina,
реальные DINO-признаки, методы с GPU-весами, реальная история ≥3 дат, реальный fusion
и пределы VRAM остаются предметом приведённой выше локальной приёмки.
