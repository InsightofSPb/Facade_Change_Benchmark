# Сокращённый history-scenarios: протокол 2026-11-09

Продолжение реализации `50ff7ad` в ветке
`codex/2026-11-08-history-three-scenarios`. Новый конфиг:
`configs/2026-11-09_history_three_scenarios_reduced.json`.
Предыдущий конфиг и schema-1 runs сохраняются; миграции старых результатов нет.
Этот документ описывает реализацию, а не измеренный выигрыш истории.
Доступный поиск `master_map` не вернул запись проекта; уточнения задания
владельца служат контрактом этого изменения.

## Матрица и обучение

| Семейства | Задача | Варианты | Число |
|---|---|---|---:|
| ConvGRU, factorized spatial/temporal Transformer | direct | history, pair-only | 4 |
| Те же | actual-next | history, last-only | 4 |
| Те же | H0 | history, last-only | 4 |
| A: semantic memory | state + downstream change | learned, repeat | 2 |
| B: pair detector | latest change | history, regularized | 2 |
| C: progression proxy | annotation-derived progression | history, latest | 2 |

Всего **18 основных и контрольных обучений**, spatial `large`, seed=13,
реальные интервалы, signed differences. A/B/C используют hidden=64.
Нет medium, mean, absolute/ordinal абляций, B-pair или shuffle/repeat inference
сетки. `resolved_matrix.json` сохраняет полные IDs и план до обучения;
недостаточная разметка C означает `INSUFFICIENT_DATA`, а не замену данных.
План и фактически законченные обучения учитываются отдельно.

Основные модели: максимум 60 эпох, AdamW lr=0.0003, weight decay=0.01,
microbatch=1, накопление до effective batch=8, AMP на CUDA, grad clip=5.
Patience=12 сбрасывается только при строгом улучшении основного downstream
VAL building-macro F1. Вторичные метрики не продлевают обучение.
OOM сохраняет диагностическую ошибку; размер, разрешение и бюджет не уменьшаются.

На frozen TRAIN-картах 16 вариантов (12 основных + A/B, без C) обучаются
LogReg (3 параметра), Score-MLP 2→32→1 GELU (129), Score-CNN
3×3(2,16)→GELU→3×3(16,16)→GELU→1×1(16,1) (2641): **48 голов**.
Основной сигнал actual-next/H0 — существующее преобразование U;
residual отдельно сохраняется и оценивается, без второго комплекта голов.

Четыре H0-варианта дополнительно получают Feature-MLP и Cross-Attention:
**8 голов**, всего **56 коротких обучений**, 74 обучаемых варианта.
Обе feature-головы видят одинаковые TRAIN-нормализованные текущие DINO features,
каузально предсказанное среднее, геометрическую поддержку, stack и явные 2D
координаты. Прогноз H0 вычисляется до доступа downstream-головы к текущему кадру.
Attention: d=64, 4 heads, один блок, FFN=128, residual/LayerNorm, 256 токенов
16×16 на полный кроп 224×224; keys и queries маскируются, пустая поддержка конечна.
Feature-MLP: две проекции d=64 и pointwise hidden=128.
При C=384 параметров соответственно 66 308 и 91 588; это не равные compute.
FLOPs без измерения помечены `NOT_MEASURED`.

Fusion: Adam, lr=0.001, weight decay=0, masked BCE, равный вклад физических
зданий, 5 эпох по умолчанию. Явная настройка допускает 1–10 эпох без поиска.
Эпоха выбирается по VAL F1 в точке FPR≤0.05; при равенстве — более ранняя.
Базовые сети не переобучаются и не запускаются заново на каждой эпохе головы.
GT/known masks влияют только на loss и оценку, не являются входными признаками.

References без обучения новых сетей: stack на общей поддержке,
stack-original на исходной поддержке (отдельная диагностика), два copy-last
с TRAIN-fitted ошибками для actual-next/H0 и independent semantic state A.

## Порог, checkpoints, resume

Порог выбирается на VAL: максимальный building-macro recall при FPR≤0.05;
при равенстве — меньший FPR, затем больший порог. После выбора порога эпохи
ранжируются по F1; равный F1 разрешается более ранней эпохой.
Сохраняются top-3 и лучшие IoU/recall, применимый NLL или semantic mIoU A.
Один уникальный state_dict хранится одним файлом с несколькими причинами отбора.
Дополнительные победители содержат веса и контракт восстановления; полный
TRAIN/VAL/TEST-export выполняется только для основного checkpoint метода.

Последний resume-checkpoint дополнительно содержит optimizer/scheduler/scaler,
Python/NumPy/Torch/CUDA RNG и прогресс. Payload записывается атомарно и проверяется;
журнал транзакции восстанавливает прерывание между записью и сменой указателей.
Победители независимы от latest. Удаляются лишь заменённые собственные временные
артефакты после commit указателей, с явным retirement в append-only `events.jsonl`.
Нельзя проверять уже retired файл как обязательный живой артефакт.

## Научные массивы и порядок операций

Schema 2 использует существующий RunStore: небольшие JSON descriptors и
content-addressed exact-dtype NPY. Большие поля читаются выборочно через read-only
memmap; проверка хеша выполняется при открытии и при изменении stat, а не на
каждом шаге головы. Нет платформы хранения или изменения старых NPZ runs.

Общие targets/support/unknown, классовые targets, RGB/семантические входы,
нормализованные DINO targets и геометрия сохраняются один раз либо имеют
проверяемые source references. Mean/scale принадлежат одному model/checkpoint/event;
residual/fusion ссылаются на них. Состояния A имеют ключ
model/checkpoint/track/observation/prefix contract. Float maps и диагностики
остаются float32 или исходного точного dtype. Нет fp16-компромисса, JPEG численных
результатов, attention matrices, autograd graphs или всех скрытых активаций.

Score fusion: сначала canonical overlap merge исходных stack-компонент
(AnyChange OR) и score метода, затем frozen stack, clipping/logit и голова.
Для CNN блок получает halo=2; loss и экспорт обрезают только центр после головы.
Недоступные входы обнуляются после logit по геометрии. Известность GT не меняет
вход головы. Нелинейность нельзя переставлять с объединением перекрытий.
Feature-MLP/Attention обе работают на полной native-grid кропа, затем sigmoid,
bilinear resize с align_corners=False, затем арифметическое объединение перекрытий.
Схемы и точные recipes записываются в contracts; при невозможности точной
реконструкции сохраняется нужный raster.

Сохраняются все TRAIN/VAL/TEST события, включая ошибки модели и неизвестные
области. Галерея — предварительный просмотр; индекс всех событий и
`report-only --event-id ID` дают доступ к любому сохранённому событию.
`--method ID` можно повторять. Report-only читает сохранённые данные без обучения,
основных моделей и декодирования всего исходного датасета. Отсутствующая
обязательная зависимость вызывает явную ошибку. Внешние кэши перечислены в
`2026-11-08_storage_dependencies.json`; их нельзя изменять или удалять.

## Preflight перед full

Проверяются manifests, ontology/channels, здания и временные интервалы,
checkpoint provenance, формы/длины и кэши, свободные RAM/VRAM/disk и процессы GPU.
Профиль использует детерминированные TRAIN typical/long/max истории, warmup,
несколько шагов, фактическое accumulation/AMP и CUDA synchronization.
Модели временные; RNG и normalizer основного обучения от профиля независимы.
Научные TEST predictions/targets/metrics для измерения ресурсов не используются.

Отдельно записываются наблюдаемые первый/повторный IO (без очистки page cache),
forward/backward/optimizer, VAL inference/merge/calibration/metrics,
scientific export/checkpoint write, пять видов fusion, saved-map replay,
bootstrap и визуализация. Оценки полного лимита эпох отделены от измерений;
неизмеренные части отмечаются явно. Записываются allocated/reserved VRAM,
host RSS, тестовая запись, постоянный/временный объём и существующие input caches.
Сжимаемость случайной модели не принимается за сжимаемость обученной.
Ранее обсуждавшиеся 58 GiB payload / 70–80 GiB workspace для 2089 событий
не являются лимитом или гарантией: объём пересчитывается по текущей схеме.

JSON/text profile содержит fingerprint code/config/data/hardware. Перед full
профиль проверяется и свободные ресурсы измеряются повторно. Без совместимого
профиля `full` выполняет только preflight и печатает команду продолжения.
Недостаток ресурсов не вызывает удаления результатов или уменьшения бюджета.
В Cloud нет реальных данных/CUDA: соответствующие результаты `NOT_MEASURED`;
CPU smoke доказывает только работоспособность реализации.

TTY использует tqdm, перенаправленный лог — обычные периодические строки
с этапом/моделью/эпохой/числом событий/rate/ETA/loss. Прошлая VAL-метрика
подписывается своей эпохой; исключения и resume/skips отражаются в ledger.

## Интерпретация таблиц

Главная таблица выбирает на VAL по одной representation/fusion/score строке
на каждый из 9 основных history-вариантов и необходимые baseline-строки.
Residual участвует в выборе. Все 74 обучаемых варианта и standalone U/residual
остаются в подробных результатах. Выбор по TEST запрещён.

Отдельные поля отвечают на два вопроса: вклад истории относительно обученного
контроля с той же головой и представлением; практический эффект относительно
stack на общей поддержке. Missing control не даёт положительного вывода об
истории. Для attention есть дополнительное сравнение с Feature-MLP с теми же
входами. C сравнивается с latest на собственной proxy-цели, stack=N/A.
Semantic state A выводится отдельно от downstream change.

Практическая значимость: ΔF1≥0.02 или ΔIoU≥0.015 при ΔFPR≤0.005 и потере
recall≤0.01; либо относительное снижение FPR≥20% при потере recall≤0.01
и F1≤0.005. Величина эффекта и paired building-bootstrap CI — разные поля.
Для прогнозного вывода дополнительно нужен меньший NLL основного checkpoint
против trained last-only и copy-last на одинаковой области/quantizer.
NLL называется model bits/component, не размером сжатого bitstream.
TEST ранее просматривался: выводы exploratory, bootstrap не оценивает seeds.

## Команды владельца

Пути источников взяты из нового конфига без изменения:
`2026-11-04-s192-full-001`, `2026-11-06-s192-fusion-001`,
`2026-11-08-s192-memory-001` и прежние memory-evaluate/followups runs.
Использовать новую выходную директорию; не пересекать её с источниками.

```bash
git -C /home/sasha/Facade_Change_Benchmark fetch origin codex/2026-11-08-history-three-scenarios
git -C /home/sasha/Facade_Change_Benchmark switch codex/2026-11-08-history-three-scenarios
git -C /home/sasha/Facade_Change_Benchmark pull --ff-only origin codex/2026-11-08-history-three-scenarios
```

```bash
cd /home/sasha/Facade_Change_Benchmark
conda run --no-capture-output -n scd_bench python -s -u -m data_analysis history-scenarios preflight --config /home/sasha/Facade_Change_Benchmark/configs/2026-11-09_history_three_scenarios_reduced.json --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-09-history-reduced-001
```

```bash
cd /home/sasha/Facade_Change_Benchmark
conda run --no-capture-output -n scd_bench python -s -u -m data_analysis history-scenarios smoke --config /home/sasha/Facade_Change_Benchmark/configs/2026-11-09_history_three_scenarios_reduced.json --device cpu --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-09-history-reduced-smoke-001
```

```bash
cd /home/sasha/Facade_Change_Benchmark
conda run --no-capture-output -n scd_bench python -s -u -m data_analysis history-scenarios full --config /home/sasha/Facade_Change_Benchmark/configs/2026-11-09_history_three_scenarios_reduced.json --preflight-report /home/sasha/Facade_Change_Benchmark/runs/2026-11-09-history-reduced-001/2026-11-08_resource_report.json --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-09-history-reduced-001
```

```bash
cd /home/sasha/Facade_Change_Benchmark
conda run --no-capture-output -n scd_bench python -s -u -m data_analysis history-scenarios resume --config /home/sasha/Facade_Change_Benchmark/configs/2026-11-09_history_three_scenarios_reduced.json --preflight-report /home/sasha/Facade_Change_Benchmark/runs/2026-11-09-history-reduced-001/2026-11-08_resource_report.json --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-09-history-reduced-001
```

```bash
cd /home/sasha/Facade_Change_Benchmark
conda run --no-capture-output -n scd_bench python -s -u -m data_analysis history-scenarios report-only --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-09-history-reduced-001
```

Основные артефакты относительно `--out`:

| Результат | Путь |
|---|---|
| Главная таблица | `2026-11-08_main_table.csv` и `.json` |
| Подробные строки/метрики | `2026-11-08_detailed_table.csv`, `2026-11-08_metrics.csv` |
| Buildings/paired CI/prognosis/A | `2026-11-08_per_event.csv`, `2026-11-08_per_building.csv`, `2026-11-08_comparisons.csv`, `2026-11-08_prediction_metrics.csv`, `2026-11-08_semantic_metrics.csv` |
| Все карты и ссылки | `2026-11-08_maps_manifest.json`, `arrays/`, `2026-11-08_storage_schema.json` |
| Primary checkpoints | `models/<method>/` |
| Fusion checkpoints/контракты | `fusion/` |
| Ресурсы | `2026-11-08_resource_report.json`, `.txt` |
| Протокол и аудит | `2026-11-08_resolved_matrix.json`, `2026-11-08_resolved_config.json`, `events.jsonl`, `run.json` |
| Галерея | `2026-11-08_gallery/2026-11-08_index.html` |

Фактические Cloud-проверки и ограничения фиксируются отдельно в
[validation](2026-11-09-history-three-scenarios-reduced-validation.md).
