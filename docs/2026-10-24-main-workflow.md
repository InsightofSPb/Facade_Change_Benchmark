# Указатель компонентов и рабочих сценариев

Актуализирован 2026-11-17 для `dev_late`. Научные решения находятся в
[master-map](2026-11-17_dissertation_research_master_map.md), правила изменений —
в [AGENTS.md](../AGENTS.md). Здесь выбирают владельца операции и его runbook.
Дата в имени исходника не определяет его актуальность. Для активной работы
используйте `dev_late`; команды переключения на прежние ветки в датированных
протоколах относятся к воспроизведению их исходных запусков.

## Владельцы операций

В командах `workflow` означает `python -m facade_change workflow`.
Пути конфигураций ниже — примеры формата; личные пути и веса задаются явно.

| Направление / операция | Канонический владелец и потребители | Команда / конфигурация | Проверки | Подробная инструкция |
| --- | --- | --- | --- | --- |
| D: COCO, идентичность, split | `data.py` читает COCO; `dataset_setup.py` и `preparation.py` готовят manifest/split; потребители — semantic dataset и crop export | `workflow prepare dataset`; `configs/2026-10-03_dataset.example.json`, `2026-10-03_metadata_rules.json` | `test_dataset_setup.py`, `test_facade_split_v4.py`, `test_semantic_leakage.py`, `test_dataset_extension_guard.py` | [Подготовка](2026-10-11_two_path_dataset_preparation.md) |
| D: геометрия и исходный RGB | `alignment.py`, `geometry.py`, `alignment_session.py`; потребители — batch, crop/history export, agentic pipeline | `workflow prepare align/crops`; параметры выбранной сессии, `--alignment-session` | `test_alignment_session.py`, `test_batch.py`, `test_crop_dataset.py`, `test_core.py` | [RGB ниже](#rgb-baseline), [истории](2026-10-25_history_features.md) |
| D/V: истории и признаки | `history_features.py`; потребители — history models, spatial change, agentic history | `workflow prepare crop-histories/history-features`; `configs/methods.example.json` | `test_history_features.py`, `test_temporal_inputs.py` | [Контракт истории](2026-10-25_history_features.md) |
| D/X: входы пар и целевые маски | `annotation_targets.py`, `crop_benchmark.FrozenCrops/ExportCrops`; потребители — crop benchmark и history targets | `workflow prepare targets`, `workflow benchmark crops`; `--targets-run`, `--branches`, явный источник | `test_2026-10-25_history_targets.py`, `test_hypothesis_dataset.py`, `test_alignment_session.py` | [Цели](2026-10-25_history_targets.md), [парный протокол](2026-10-25-crop-benchmark.md) |
| V: детекторы и модели | `methods/registry.py`, `methods/base.py`, `methods/remote.py`; потребители — benchmark, real history и `agentic/pipeline.py` | `workflow benchmark synthetic`; `configs/methods.example.json`, `benchmark.example.json` | `test_method_registry.py`, `test_method_workers.py`, `test_common_runner.py` | [Общий runner](2026-10-04_common_runner.md) |
| V: состояние одного кадра | `segmentation/` и `semantic_training.py`; semantic routes используют адаптеры `scripts/real_eval/semantic_adapters.py` | `workflow train segmentation`, `workflow benchmark segmentation --evaluate-run …` | `test_segmentation_metrics.py`, `test_segmentation_train.py`, `test_semantic_adapters.py` | [Сегментация](2026-10-11_dino_semantic_training.md) |
| V: парные метрики и агрегация | `hypothesis_benchmark.py` — synthetic H0/H1; `real_gt_metrics.py` — бинарные counts/метрики реальных переходов; `real_gt_slices.py` вызывает те же формулы и добавляет классовый ROI; `crop_benchmark.Macro` — crop protocol | `workflow benchmark synthetic/crops/metrics`; разные явные protocols | `test_hypothesis_benchmark.py`, `test_real_gt_metrics.py`, `test_real_gt_slices.py`, `test_workflow.py` | [H0/H1](2026-10-04_h0h1_benchmark.md), [реальные метрики](2026-10-17_real_eval_tools.md) |
| V: PREVIOUS/EARLIEST/HISTORY | `research/2026-11-04_history_benchmark.py`; dispatch из `scripts/real_eval/real_temporal_night.py` | `workflow prepare real`, `workflow benchmark pilot/real`; frozen prepared manifest | `test_2026-11-04_history_benchmark.py`, `test_temporal_history.py` | [Канонический benchmark](2026-11-04-canonical-history-benchmark.md) |
| V: анализ сохранённых карт | `data_analysis/2026-11-06_history_fusion.py`, `2026-11-08_memory_evaluate.py`, `2026-11-08_memory_followups.py`; transition bank принадлежит `research/2026-11-08_transition_bank.py` | `python -m data_analysis history-fusion/memory-evaluate/memory-followups --help` | `test_data_analysis.py` подключает датированные component tests | [Fusion](2026-11-06-history-fusion.md), [память](2026-11-08-variability-memory.md), [follow-ups](2026-11-08-memory-followups.md) |
| V: три сценария истории | `research/2026-11-08_history_scenarios_*`; один CLI `data_analysis/2026-11-08_history_scenarios.py` | `python -m data_analysis history-scenarios`; `configs/2026-11-08_history_three_scenarios.json` и reduced config | `test_2026-11-08_history_scenarios_*.py` через `test_data_analysis.py` | [Основной опыт](2026-11-08-history-three-scenarios.md), [reduced-профиль](2026-11-09-history-three-scenarios-reduced.md) |
| D/V/R: run, cache и provenance | `io.py` — файлы/hashes; `history_features.Fingerprints` — входы; `2026-10-14_temporal_experiment.RunStore` — журнал; `agentic/runner.py` адаптирует его, `agentic/store.py` хранит предметные наблюдения | явные `--out`, source/config/model hashes; resume только у backend | `test_temporal_experiment.py`, `test_benchmark_results.py`, `test_2026_11_08_agentic_store.py`, history storage tests | [RunStore](2026-10-14_temporal_experiment.md), [agentic](2026-11-08-change-agentic-rag.md) |
| R: визуальный инструмент | `agentic/tools.ToolService`, `agentic/pipeline.prepare_case`; те же RemotePool/registry и geometry, что у benchmark | `workflow agentic run/history/check`; `configs/2026-11-08-agentic.example.json` | `test_2026_11_08_agentic_pipeline.py`, `test_2026_11_08_agentic_tool_regressions.py`, `test_2026_11_08_agentic_support.py` | [Pair/history tools](2026-11-08-change-agentic-rag.md) |
| R: архивный ответ и exposure | `agentic/2026-11-15_www_archive.py`, `2026-11-15_www_runner.py` (`_expose_archive`, `_condition`); evidence → message blocks → финальный ответ | `workflow agentic www`; публичный questions/index и protocol | `test_2026_11_15_www.py`, `test_2026_11_15_www_smoke.py`, `test_2026_11_17_www_action_debug.py` | [WWW](2026-11-15-www-qa.md); [review](2026-11-16-www-rgb-qa-policy.md), [выбор действия](2026-11-16-www-tool-selection.md) |
| R: inspection и общие контракты | `agentic/2026-11-10_inspection*`; identity validator — `2026-11-16_www_identity.py`, inspection задаёт pending-policy; строгий JSON — `io.strict_json_decoder`, схемы ответов остаются разными | `workflow agentic inspection`; `configs/2026-11-10-inspection-protocol.example.json` | `test_2026-11-10_inspection.py`, `test_2026-11-16_inspection.py`, resource verification через `test_workflow.py` | [Inspection](2026-11-10-adaptive-visual-inspection.md), [подготовка cohort](2026-11-16-cvpr-smoke-preparation.md) |
| D/V/R: галереи | `gallery/`, `visualization.py`, `agentic/report.py`; launcher-ы лишь передают аргументы | `workflow gallery …`; явный сохранённый run/bundle | `test_saved_galleries.py`, `test_real_temporal_report.py`, `test_2026_11_08_agentic_report.py` | [Atlas](2026-10-18-facade-atlas.md), соответствующий runbook сценария |

Пути без каталога относятся к `facade_change/`; real_gt-модули —
к `scripts/real_eval/`, тесты — к `tests/`. Наличие части evaluators в scripts
сохраняет действующие контракты frozen runs; их имена не означают дубли алгоритма.

## RGB baseline

После подготовки reviewed RGB crop dataset:

```bash
python -m facade_change workflow prepare synthetic \
  --crop-run /absolute/path/to/crops \
  --config configs/2026-10-04_h0h1_controls.json \
  --out runs/2026-11-17-controls-001
python -m facade_change workflow benchmark synthetic \
  --dataset-run runs/2026-11-17-controls-001 --methods rgb_diff ssim \
  --out runs/2026-11-17-baseline-001
```

Результат: `run.json`, `metrics.json`, `summary.json`,
`threshold_selection.json`, `gallery.html`, карты и маски.
`gallery comparison` ожидает фиксированный набор из шести методов и не является
report-командой для сокращённого RGB/SSIM запуска. Его контракт —
[data_analysis/README](../data_analysis/README.md).
Без локальных данных можно начать с CPU smoke из README.

## Данные и сегментация

`prepare dataset` принимает COCO и RGB. `prepare align` читает явный manifest;
`prepare crops --alignment-session …` импортирует завершённую сохранённую сессию.
Он проверяет hashes/review/split и не выбирает самый новый каталог.
`--quality-report` допускается только по проверяемому контракту импорта;
его наличие не переписывает статус исходного preparation.

`data/dataset_statistics.py` — недоступная Cloud локальная разработка владельца.
Её старая версия из `b5bdb7b8` не переносится. Активный код читает формат сессии,
а не импортирует Python-файл из старого worktree.

Обучение: `prepare augmentations` → `train segmentation`.
Оценка: `benchmark segmentation --evaluate-run …`; TEST указывается явно.
Готовая модель пропускает train. Ни benchmark, ни gallery не должны неявно
запускать обучение. Точные параметры показывает `STAGE TASK --help`.

## Истории и оценка

`prepare crop-histories` → `prepare history-features` → `prepare history-targets`
сохраняет даты, интервалы, поддержку и происхождение целей. Неизвестная область
не становится отрицательной; annotation-derived переходы не объявляются gold.

Канонический history benchmark выбирает пороги по VAL. Парный `benchmark crops`
сохраняет исторический clean-TRAIN threshold protocol. H0/H1 synthetic,
LOFO, segmenter metrics, supplied-region QA и archive answer metrics имеют
собственные популяции и правила нулевых знаменателей; одно имя F1 их не объединяет.

Для `history-scenarios` режимы `preflight`, `smoke`, `full`, `resume`, `report-only`
принадлежат одному backend. `full` включает обучение и требует его preflight.
Конфигурации полного и reduced опыта задают параметры общего кода.
Существующие публичные config содержат исторические пути: для нового реального
запуска задайте свой явный локальный JSON. Старый config остаётся частью provenance.
`history_scenarios_storage.SharedMapStore` — индекс карт и memmap,
а не второй журнал запусков; lifecycle использует существующий RunStore.

## Агентские сценарии

```bash
python -m facade_change workflow agentic --help
python -m facade_change workflow agentic qa --help
python -m facade_change workflow agentic www --help
python -m facade_change workflow agentic inspection --help
```

Эти команды передают аргументы существующему `agentic.cli`.
Для проверки «нашёл, но не показал» начните с `_expose_archive` и `_condition`:
сопоставьте retrieved IDs, image budget, exposure-записи и фактические message
blocks, затем citations ответа. Найденная запись не подтверждает показ её пикселей.
Inspection использует собственные action/answer schemas и strict fence policy;
общий JSON decoder не исправляет смысл ответа.

`inspection preflight` проверяет входы без inference. `agentic check` проверяет
ресурсы без загрузки моделей, пока не передан `--load-model`.
`workflow --dry-run` показывает dispatch и аргументы без импорта backend;
отсутствующие файлы в таком плане допустимы, готовность проверяет backend.
Последовательный `workflow run --config` не добавлен: рабочие сценарии уже имеют
свои фазовые контракты, а единого проверенного формата передачи результатов нет.

## Добавить метод или датасет

Метод: реализуйте callable-адаптер по `methods/base.py`, зарегистрируйте его в
существующем `methods/registry.py` и задайте options в methods config.
Вход — исходный RGB и geometric support; выход — карта на той же решётке,
`output_kind`, metadata и provenance. Native binary mask не подменяется score.
Проверьте fake/alternate adapter в `test_method_registry.py` и сквозной runner.
Новый метод не копирует подготовку данных, runner или evaluator.

Датасет: сначала выберите задачу и существующий контракт — COCO/semantic,
reviewed RGB crops, augmented H0/H1 или canonical histories. Адаптер переводит
формат в этот контракт, сохраняя sample/building IDs, split, время, геометрию,
метки и происхождение. Для парных экспортов используйте seam
`crop_benchmark.ExportCrops`, для аугментаций — `FrozenCrops`; внешний формат
не должен выдавать себя за существующий kind без необходимых полей и hashes.
Проверки независимого источника/подменяемого input adapter уже есть в
`test_alignment_session.py`, `test_real_gt_metrics.py` и `test_real_robustness_prepare.py`.
Не выдумывайте даты, историю или pixel gold для image-level набора.
Внешние научные датасеты этим рефакторингом не загружались и не валидировались.

## Пять коротких маршрутов чтения

| Вопрос | От стартовой страницы к проверке |
| --- | --- |
| Простой baseline? | README → RGB baseline выше → `hypothesis_benchmark.py` → `test_workflow.py` |
| Новый датасет? | README → «Добавить метод или датасет» → выбранный provider/manifest → contract fixture |
| Метрики и агрегация? | README → строка своего протокола в таблице → evaluator/Macro → его тесты |
| Инструмент нашёл фото, но модель его не видела? | README → «Агентские сценарии» → WWW exposure/message blocks → WWW/inspection regressions |
| Пересечения зданий и будущие наблюдения? | README → D split / R tools → `preparation.py`, `segmentation/data.py`, `agentic/store.py` → leakage и agentic store tests |

## Совместимость и история

Изменён только рекомендуемый вход агентских сценариев:
`python -m facade_change.agentic …` → `python -m facade_change workflow agentic …`.
Старый вход сохранён. `workflow benchmark agentic`, `prepare agentic-index` и
`gallery agentic` остаются совместимыми узкими маршрутами.

Исторические команды, отчёты переноса и проверки от 24 октября находятся
в [исходной редакции документа при bf4f4c4](https://github.com/InsightofSPb/Facade_Change_Benchmark/blob/bf4f4c47e321259a8023812776c21252c006fd27/docs/2026-10-24-main-workflow.md).
Точные старые результаты воспроизводятся на указанном в их provenance commit;
рефакторинг меняет source hashes даже при численном паритете.
[Аудит dev_late](2026-11-17-dev-late-audit.md) фиксирует переносы и измеренные проверки.
