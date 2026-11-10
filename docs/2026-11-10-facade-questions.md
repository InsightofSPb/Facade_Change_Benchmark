# Вопросы по истории фасада: курация и офлайн-оценка

Дата: 2026-11-10. Ветка: `codex/2026-11-08-change-agentic-rag`.
Вход: `python -m facade_change.agentic qa --help`.

Протокол **supplied-region QA**: участок уже задан, в том числе с использованием
аннотаций. Метрики не описывают поиск изменившихся участков на всём фасаде.
Новые команды работают на CPU, без моделей, alignment, детекторов и эмбеддингов.
Существующие `run`, `history`, `compare`, модели и их бюджеты сохранены.

## Входы и генерация

| Вход | Контракт |
|---|---|
| Каталог `--histories` | `manifest.json`, `kind=canonical_crop_histories`, постоянные track/building/view/split, наблюдения, RGB/support, сохранённые матрицы, завершённый immutable report |
| Необязательный `--annotation-targets` | `history_annotation_targets/index.json`, привязанный к SHA именно этого manifest; онтология/category_to_channel; `packed:uint16`, `support:bool` на той же решётке |
| `--knowledge-cutoff` | Предел доступности знаний, отдельный от дат снимков |
| `--task-mode` | `retrospective` либо `as_of` |
| `--available-at` | Дата доступности архива либо явное допущение `observation_date` |

Форматы: [истории](2026-10-25_history_features.md),
[перенос аннотаций](2026-10-25_history_targets.md). NPZ переиспользуются, новой
растеризации COCO нет. Без карт предлагаются uncertain controls. Для раннего
свидетельства по умолчанию используется существующая группа `structural_damage`;
`--damage-categories NAME ...` задаёт имена из сохранённой онтологии. При поиске
переходов рассматриваются все сохранённые категории, включая архитектурные элементы.

Детерминированные русские шаблоны:

* Повреждение размечено в первом геометрически пригодном наблюдении:
  «Какое самое раннее датированное свидетельство повреждения этого участка доступно?»
  Provisional answer — «уже присутствовало к 2009», **не** «возникло в 2009».
  Физическая видимость ещё требует проверки человеком.
* Для соседних наблюдений разных дат: «Что изменилось в этом участке между
  наблюдениями, если вообще изменилось?» Пространственное различие битовых карт
  в известных пикселях одного участка создаёт `possible_change`. Расширение
  повреждения при сохранении класса тоже обнаруживается как кандидат.
* Полное совпадение аннотаций на геометрической области создаёт только
  `stable_control`; неполнота разметки/покрытия — `uncertain_control`.

Сравнение по embedding similarity или агрегированным числам классов не применяется.
Аннотации — сигнал отбора, не временной GT, причинная цепочка деградации или
доказательство отсутствия неразмеченного объекта. Исчезновение архитектурного
элемента в частично размеченной области может потребовать uncertain review.

Все строки начинают со статуса `candidate`, visibility=`unknown`. Сохраняются
ID, даты, геометрия, SHA, причины отбора, provisional answer, RGB и приватные
annotation previews. Оранжевый в preview — выбранные повреждения, синий — другая
известная разметка, чёрный — неизвестность.

Наблюдения, датированные или доступные позже cutoff, исключаются с причиной в
`skipped`. `task_dates`, `available_at`, `knowledge_cutoff` раздельны. Подтверждённые
годы берутся из существующего canonical-history контракта, не из имени файла.
Общая дата available_at — явное допущение запуска; смешанные архивные даты пока
нужно подготовить отдельно. Runner chronology не изменена. Снимки одного года
не становятся временным переходом; генератор не объединяет их в доказанное
состояние. Неоднозначные альтернативы проверяет рецензент.

## Курация и версии

`generate` создаёт приватные `candidates.json`, `review.csv`, `gallery.html`,
`review_template.json`. CSV/HTML — для просмотра. **Скопируйте** JSON-шаблон за
пределы immutable каталога и редактируйте копию. Изменять записанные кандидаты,
previews и шаблон на месте нельзя: последующие команды проверяют их SHA.

Каждая изменённая строка требует `reviewer`, обоснование в `private_notes`,
`expected_revision` (сначала 0), `status`: `candidate`, `approved`, `rejected`.
Неизменённые пустые строки пропускаются. Только approved попадает в scored cohort.
При пересмотре передайте `--previous` и текущую revision: создаются новый каталог,
цепочка ссылок с SHA и событие `qa_review_decision`. Старые файлы не перезаписываются;
устаревшая revision отклоняется.

Необязательный `reviewed_question` задаёт непустую человеческую формулировку.
Original candidate/question остаётся неизменным, ID вопроса сохраняется. При первом
review без override используется шаблон; в последующих ревизиях отсутствие поля
сохраняет ранее проверенный текст. Чтобы восстановить шаблон, явно передайте его
исходный текст. CSV/HTML показывают действующую формулировку, а публичный экспорт —
её в обычном поле `question`, без приватных полей review. Изменение формулировки
меняет SHA публичного manifest: старые прогнозы не подходят к новому экспорту.

Минимальная строка decisions для редактирования ещё не утверждённого кандидата
(ID и revision берутся из ваших файлов; это не утверждение реального GT):

```json
{
  "question_id": "existing-question-id",
  "expected_revision": 0,
  "status": "candidate",
  "reviewer": "local-reviewer",
  "private_notes": "Уточнена формулировка; ответ ещё требует проверки.",
  "reviewed_question": "Какие видимые различия есть между наблюдениями этого участка?"
}
```

Строка помещается в `decisions` скопированного шаблона с его `candidate_sha256`.
Для approved по-прежнему нужен проверенный label. Неизвестные поля строки review
отклоняются: ошибочный ключ `question` даёт подсказку использовать `reviewed_question`.
Пустой override тоже является ошибкой, а не неявным сбросом текста.

Пример `label` для видимого изменения. **Замените ID и координаты проверенными
значениями своего кандидата** — пример не является готовой разметкой:

```json
{
  "assessment": "confirmed_change",
  "observability": "visible",
  "reference_answer": "Краткое проверенное описание видимого различия.",
  "limitations": [],
  "archival_status": "unknown",
  "supporting_evidence_ids": [],
  "events": [{
    "event_id": "reviewed-physical-episode-id",
    "coordinate_system": "region_crop_pixels",
    "region_id": "existing-track-id",
    "bbox_xyxy": [10, 20, 30, 40]
  }]
}
```

Один физический эпизод в перекрывающихся кропах получает один reviewed `event_id`
и отдельные рамки в каждой решётке. Разные эпизоды не объединяются. ID не может
охватывать разные здания или временные интервалы. Bbox — конечные
`[left, top, right, bottom]` положительной площади внутри reference-кропа.
Native-координаты нельзя переименовывать в crop: требуется явный проверенный перенос
до курации. Новый evaluator преобразования не угадывает.

| `assessment` | Смысл |
|---|---|
| `confirmed_change` | Различие оценивается по разрешённым снимкам; нужны `events` |
| `confirmed_stability` | Стабильность проверена; `events: []`; это не «неповреждённый фасад» |
| `insufficient_evidence` | Непустой `limitations`, `events: []`; hidden state не входит в eligible GT |
| `earliest_presence` | Только для раннего свидетельства: `earliest_observation_id`, `present_by` с его датой, `onset: "unknown"`, `events: []` |

`observability`: `visible`, `partly_visible`, `occluded`, `missing_coverage`,
`poor_quality`, `other`, `unknown`. Полностью закрытый/непокрытый участок не получает
assessable verdict. Неизвестная видимость сама по себе не запрещает подтвердить
отчётливо видимое различие. Geometric support не является маской видимости.

Отдельно известное скрытое состояние хранится только в приватном `underlying_state`,
например с `assessment`, `evidence_ids`, `source.path/sha256` другого снимка и
`not_permitted_to_system: true`. Оно не участвует в GT для закрытого участка;
обоснованное воздержание не штрафуется. Связь дополнительных свидетельств проверяет
человек: автоматического разрешения архивных ID пока нет.

`archival_status`: `potentially_useful`, `verified_useful`, `unavailable`, `unknown`.
Для verified_useful требуются приватные `supporting_evidence_ids`. Ранний вопрос
сам по себе не доказывает полезность архива.

Для действительной временной маски добавьте в event `mask_path`, `mask_sha256`,
`mask_task: "temporal_change"`. Бинарные PNG/NPY должны совпадать с решёткой reference,
быть непустыми, внутри рамки и фиксированной ранней геометрической поддержки.
Семантическая маска одного снимка для этого непригодна.

## Публичные вопросы и приватные ответы

`export` создаёт два файла:

* `questions.json`: whitelist вопросов, building/view/region/split, observation IDs,
  дат/cutoff, RGB/support путей и SHA, координат и сохранённых преобразований.
* `answers.private.json`: ответы, reviewed events/masks, observability, ограничения,
  reviewer/revisions и supporting evidence IDs. Привязан SHA к точному questions.json;
  обратной ссылки в публичном файле нет.

Модели/индексу передаются **только** публичный manifest и разрешённые материалы.
Не индексируйте весь каталог export/review: соседние JSON/CSV/HTML, previews и
журналы приватны. Приватные JSON имеют `access: private_evaluation`. Online guard
отвергает marker, поля ответов/аннотаций/курации во вложенных структурах и
сериализованных JSON; проверяется и контекст перед model messages. Ручной пересказ
закрытой информации без этих полей требует дисциплины подготовки корпуса:
это не система разграничения прав файловой системы.

`retrieval_contract` фиксирует `jina-clip-v2`, естественные единицы
`dated_observation`/`archival_record`, варианты photos_only, multimodal_retrieval,
graph_retrieval, bounded_agentic_retrieval. Произвольного chunking нет.
`implemented_systems: []`: четыре системы здесь не реализованы. Существующий Jina
adapter не заменён. Автоматический запуск нового question manifest через runner
ещё не подключён; подготовлены общие данные и офлайн-вход для сохранённых ответов.

## Прогнозы и метрики

`qa evaluate` принимает `facade_qa_predictions` с SHA точного публичного manifest.
Системы должны оцениваться на одной frozen approved cohort. Wrapper для старого
результата (здесь условные пути/ID):

```json
{
  "schema_version": 1,
  "kind": "facade_qa_predictions",
  "model_manifest_sha256": "sha256-of-questions-json",
  "ranking_source": {"kind": "single_box", "source": "saved agentic result.json"},
  "cases": [{
    "question_id": "existing-question-id",
    "result_path": "/actual/saved/result.json",
    "result_sha256": "sha256-of-result-json"
  }]
}
```

Для нескольких сохранённых регионов вместо result_path задайте `status`, `decision`,
`regions`. В регионе нужны `prediction_id`, bbox и владелец координат, а также
`rank` при `ranking_source.kind=saved_rank` либо конечный `score` при
`score_descending`. Ранги уникальны; ties score разрешаются по prediction ID.
`source` документирует происхождение порядка, без использования GT. Можно
переопределить ranking_source у отдельного вопроса.

`status`: completed/abstained/error/failed/missing. `decision`: change/no_change;
для воздержания — insufficient_data/incomparable; для раннего свидетельства —
earliest_presence с earliest_observation_id. Отсутствующая строка означает missing.
Ошибка исполнения остаётся ошибкой, даже если рядом записано insufficient_data.
Противоречивые status/decision/regions считаются технической ошибкой. При no_change
локализация не считается детекцией. Раннее свидетельство оценивается по ID отдельно.

Отсутствующий индивидуальный result_path считается missing; нечитаемый JSON,
неверная структура результата (например, `regions: {}`) или несовпавший SHA
индивидуального результата — техническая ошибка этого вопроса. Остальные случаи
продолжают оцениваться, все approved случаи и eligible events остаются в когорте.
Диагностика в cases.csv/evaluation.json/HTML содержит question ID, путь при наличии
и причину. Ошибка не считается justified abstention. Сбой необязательной маски или
support прогноза отключает только pixel diagnostic; допустимая рамка сохраняется.

Глобальные нарушения по-прежнему останавливают оценку: mismatch manifest/cohort SHA,
дубли/неизвестные question IDs, приватные поля в manifest или загруженном результате,
недопустимая ranking policy/GT-ordering, некорректный reviewed GT или его hashes.
Граница обработки ошибок не скрывает эти нарушения. Ошибочные bbox/coordinate
owners по-прежнему учитываются как unmatched predictions с диагностикой.

Из сохранённого result.json берётся максимум одна исходная рамка; при K=3/5/10
она не размножается. Другие/неполные координатные владельцы, несовместимая решётка
и недопустимые рамки дают unmatched prediction с диагностикой. Вывод модели не
превращается в сегментацию.

1. Top K **для каждого вопроса**, K=1/3/5/10, выбирается до GT matching по сохранённому
   порядку. Показываются фактические доступные и использованные прогнозы.
2. Локализация оценивается только на assessable change/stability. Для unassessable
   отдельно считаются justified abstentions и unsupported confident decisions,
   включая уверенное no_change; скрытый underlying state не входит в eligible GT.
3. Сопоставление детерминированное, взаимно однозначное и максимальное по числу пар.
   Один прогноз не покрывает несколько событий; reviewed event ID дедуплицируется
   между кропами. Сравниваются рамки одного вопроса в одной решётке.
4. IoU 0,25 — начальный выбор для грубой локализации; 0,5 — строгая диагностика,
   не универсально валидированные пороги. `--iou 0.25 0.5` записывается в протоколе.
5. Recall = hits / eligible physical events, precision = hits / used predictions.
   Дубли не дают новых hits. Missing/error не удаляют GT из знаменателя. Без
   положительных событий recall=null; без прогнозов precision=null.
6. False-alarm rate = стабильные assessable случаи с завершённым change / все
   стабильные assessable случаи фиксированной когорты. Читать вместе с coverage:
   missing не считается доказанной стабильностью.

Переиспользованы существующие box IoU, coordinate-owner checks и pixel metrics.
Для WWW pixel diagnostics явно применяется политика `fixed_target`: область оценки
берётся из immutable reference_support исходного кандидата и проверенной review,
независимо от прогноза. Проверяются цепочка SHA, согласие экспортированных labels
с review, владелец/размер решётки, бинарные маски и фиксированная reference support.
Для этого исходные candidate/review artifacts должны оставаться доступны.
Прежний `agentic compare` сохраняет собственную политику intersection; она не меняется
молча вслед за WWW.

Prediction support должна покрывать **всю** фиксированную область. При полном
покрытии TP/FP/FN и IoU/F1 считаются на этой области. При неполном покрытии:
`status=not_available`, `reason=incomplete_prediction_support`, `required_pixels`,
`covered_pixels`, `coverage_fraction`, без IoU/F1. Пустая support не улучшает оценку.
Если temporal masks или область оценки не заданы, сопоставимых pixel metrics нет;
область не достраивается по bbox или ответу модели. Повреждение заявленных GT
артефактов/хешей остаётся фатальной ошибкой целостности, а не пропуском GT.

Геометрическая support не является физической видимостью. Этот контроль покрытия
не меняет box matching или фиксированные знаменатели когорты и change-size subsets.
Это не полный сегментационный benchmark; box-only случаи масок не получают.

Площадь изменения — пиксели reviewed temporal-event mask / **фиксированная ранняя
геометрическая поддержка reference-кропа**. Сохраняются shape, frame, region ID,
support SHA и знаменатель; поздняя окклюзия его не уменьшает. Main cohort содержит
все события. Дополнительные subsets >=1% и >=5% получают recall/counts, box-only
area недоступна. Конфликтующие reference area одного события в разных кропах
исключают только его size-stratification с явной причиной; нужна единая проверенная
опора. Полнофасадные и crop-проценты не смешиваются.

Subset hits берутся из общего one-to-one matching; subset precision не определяется,
чтобы меньшие истинные события не стали ложными тревогами. Main precision сохраняется.
Отчёт содержит pooled counts, cases, summaries по building_id и средний building
recall по зданиям с положительными событиями. Кропы не считаются независимыми
зданиями; доверительные интервалы по кропам не вычисляются.

## Проверка в Cloud и приёмка WSL

Исходная CPU-проверка реализации `6c9c0d5`:

```bash
/workspace/work/facade-venv/bin/python -s -B -m unittest discover -s tests -p 'test_2026_11_08_agentic*.py' -v
/workspace/work/facade-venv/bin/python -s -B -m unittest tests.test_2026_11_10_facade_questions tests.test_2026_11_09_qwen_diagnostic -v
/workspace/work/facade-venv/bin/python -s -B -m facade_change.agentic qa demo --out /workspace/work/2026-11-10-facade-qa-smoke-001
git diff --check
```

Результат: 100 существующих тестов, OK (один штатный skip); 15 новых QA + 6 bbox/worker
тестов, OK. Новый набор проверяет hand-counts, expansion при тех же классах, cutoff,
immutable review, утечки, actual masks, ownership, overlap dedup, ошибки и пропуски.
CLI demo завершён с JSON в stdout, приватной review-галереей и отчётом метрик;
контрольные числа приведены ниже. Help нового входа и diff check прошли.
Полный repository suite, GPU, веса и старые эксперименты не запускались.

После исправлений WWW повторены только затронутые проверки:

```bash
/workspace/work/facade-venv/bin/python -s -B -m unittest tests.test_2026_11_10_facade_questions tests.test_2026_11_08_agentic_report tests.test_2026_11_08_agentic_reasoning -v
/workspace/work/facade-venv/bin/python -s -B -m unittest discover -s tests -p 'test_2026_11_08_agentic_store.py' -v
/workspace/work/facade-venv/bin/python -s -B -m facade_change.agentic qa demo --out /workspace/work/2026-11-10-facade-qa-repair-smoke-001
git diff --check
```

Результат: 56 тестов (в том числе 24 QA) — OK; 18 store-тестов — OK, один skip из-за
отсутствия optional CPU torch. Store-тесты требуют discover для импорта общей
fixture: первоначальный пакетный вызов дал три ошибки импорта, повтор через
discover прошёл без изменения окружения. Три новые регрессии сначала воспроизвели
исходные сбои. CLI demo завершён с корректным JSON stdout и прежними контрольными
box-counts: eligible=4, matched=1, missed=3, unmatched=3 при K=10/IoU=0,25.

Pixel-регрессия 10×10: GT `[3,1,6,4]`, прогноз `[3,1,9,4]`. Полный support даёт
TP=9, FP=9, IoU=0,5. Покрытия 1/100 и 0/100 дают incomplete_prediction_support без
IoU/F1; box match остаётся тем же. Проверены смешанная когорта valid/missing/error,
повреждённые необязательные маски, фатальные GT/hash/privacy нарушения, версии
формулировок и отказ принимать прогнозы от старого публичного manifest.

Команды локальной приёмки именно этих исправлений:

```bash
cd /home/sasha/Facade_Change_Benchmark
git switch codex/2026-11-08-change-agentic-rag
git pull --ff-only
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -s -B -m unittest tests.test_2026_11_10_facade_questions tests.test_2026_11_08_agentic_report tests.test_2026_11_08_agentic_reasoning -v
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -s -B -m unittest discover -s tests -p 'test_2026_11_08_agentic_store.py' -v
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -s -B -m facade_change.agentic qa demo \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-10-facade-qa-repair-demo-001
```

Ожидаются completed, `review/gallery.html`, раздельные export manifests,
`evaluation/evaluation.json`, `evaluation/cases.csv`, `evaluation/localization.csv`
и `evaluation/gallery.html`. Каталог должен быть новым. Реальные вопросы и GT
владелец пишет и утверждает локально; синтетика не подтверждает качество модели.

Локально используйте существующее окружение. При повторе выбирайте новый суффикс
каталога: существующий out не затирается.

```bash
cd /home/sasha/Facade_Change_Benchmark
git switch codex/2026-11-08-change-agentic-rag
git pull --ff-only
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -s -B -m unittest tests.test_2026_11_10_facade_questions -v
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -s -B -m facade_change.agentic qa demo \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-10-facade-qa-demo-001
```

Ожидается completed; в demo появятся `candidates/review.csv`, `review/gallery.html`,
`export/questions.json`, `export/answers.private.json`, `evaluation/evaluation.json`,
`evaluation/localization.csv`, `evaluation/cases.csv`, `evaluation/gallery.html`.
У каждого этапа есть run.json, config, hashes и журнал. Fixture reviewer явно
называется synthetic_fixture_not_human; прогнозы сохранены без исполнения модели.

Ручная проверка: K=10, IoU=0,25 — eligible=4, matched=1, missed=3, used=4,
unmatched=3, recall=precision=0,25. Два события b1 также представлены во втором
кропе, но знаменатель остаётся 4. При K=1 — used=2, hit=1, unmatched=1.
Mask-area >=1%: 3 события/1 hit; >=5%: 1/1. Box-only event остаётся в main cohort.
Отдельно: 2 missing, 1 runtime error, 1 justified abstention, 1 unsupported confident
decision, 1 false alarm. Синтетика подтверждает реализацию счётчиков, не качество моделей.

### Реальные данные без выдуманных путей

В Cloud реальная пара histories/annotation export отсутствует. Блок запрашивает
существующие пути на WSL: **каталог** с canonical manifest.json и точный index.json
его annotation targets. Они не подставляются из примера и не создаются автоматически.

```bash
cd /home/sasha/Facade_Change_Benchmark
read -r -p 'Каталог canonical histories: ' FACADE_QA_HISTORIES
read -r -p 'Существующий history_annotation_targets/index.json: ' FACADE_QA_TARGETS
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -s -B -m facade_change.agentic qa generate \
  --histories "$FACADE_QA_HISTORIES" --annotation-targets "$FACADE_QA_TARGETS" \
  --task-mode retrospective --knowledge-cutoff 2026-11-10 --available-at 2026-11-10 \
  --splits val \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-10-facade-qa-candidates-001
cp -n /home/sasha/Facade_Change_Benchmark/runs/2026-11-10-facade-qa-candidates-001/review_template.json \
  /home/sasha/Facade_Change_Benchmark/runs/2026-11-10-facade-qa-decisions-001.private.json
```

Дата доступности здесь — явная консервативная дата сборки архива, не утверждение
о доступности в годы съёмки. Для исторического as_of нужна подтверждённая availability
и иной cutoff. Протокол настраивается на train/val; для финального TEST сначала
зафиксируйте его и используйте `--splits test` с новыми каталогами.

Откройте `2026-11-10-facade-qa-candidates-001/gallery.html`, проверьте RGB/аннотации
и заполните копию decisions JSON: reviewer, private_notes, status, label. Затем:

```bash
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -s -B -m facade_change.agentic qa review \
  --candidates /home/sasha/Facade_Change_Benchmark/runs/2026-11-10-facade-qa-candidates-001/candidates.json \
  --decisions /home/sasha/Facade_Change_Benchmark/runs/2026-11-10-facade-qa-decisions-001.private.json \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-10-facade-qa-review-001
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -s -B -m facade_change.agentic qa export \
  --review /home/sasha/Facade_Change_Benchmark/runs/2026-11-10-facade-qa-review-001/reviewed.json \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-10-facade-qa-export-001
```

Ожидаются новый reviewed.json, CSV/HTML и два раздельных manifest. Пустой шаблон
даёт 0 approved, а не автоматическое одобрение. Для следующей ревизии используйте
новые decisions/out и `--previous .../2026-11-10-facade-qa-review-001/reviewed.json`.

После независимого исполнения системы подготовьте predictions wrapper описанного
выше формата; SHA публичного файла можно получить `sha256sum`. Этот шаг пока требует
явного сопоставления question ID с результатами системы. Evaluate сам инференс
не запускает и рейтинги не придумывает.

```bash
read -r -p 'Существующий facade_qa_predictions JSON: ' FACADE_QA_PREDICTIONS
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -s -B -m facade_change.agentic qa evaluate \
  --questions /home/sasha/Facade_Change_Benchmark/runs/2026-11-10-facade-qa-export-001/questions.json \
  --answers /home/sasha/Facade_Change_Benchmark/runs/2026-11-10-facade-qa-export-001/answers.private.json \
  --predictions "$FACADE_QA_PREDICTIONS" --iou 0.25 0.5 \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-10-facade-qa-evaluation-001
```

Ожидаются evaluation.json, cases.csv, localization.csv, gallery.html и completed
run. Completed означает успешный расчёт отчёта; ошибки самой оцениваемой системы
по-прежнему отражены в coverage. Реальный benchmark, качество retrieval/Qwen,
подтверждение физических event IDs и ручной GT в Cloud не проверялись.
