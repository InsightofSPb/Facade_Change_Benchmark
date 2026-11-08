# Проверка пользы истории: три задачи и сценарии A/B/C

Дата протокола: 2026-11-08. Ветка: `codex/2026-11-08-history-three-scenarios`.
База реализации: `c31b82951e991f5ba38ce58c49b541644996324b`, включая
`memory-evaluate` и `memory-followups`. Настоящее обучение на фасадах выполняется
локально; синтетические проверки подтверждают исполнение и контракты, но не
пользу истории.

## Вопрос и границы вывода

Проверяем, даёт ли информация из прошлого практический выигрыш относительно
сопоставимого обученного контроля без полезной истории. Дополнительные параметры,
другой decoder, fusion или loss сами по себе не являются свидетельством пользы
истории. Единственный training seed — **13**. Paired bootstrap ресемплирует
здания со всеми их наблюдениями; он не оценивает вариативность обучения.

Существующий TEST уже просматривался. Все результаты этой ветки на нём —
**exploratory**, даже при доверительном интервале, исключающем ноль.
Старые ориентиры F1 не подставляются в новые таблицы: stack и прежние результаты
проверяются по сохранённым артефактам и хешам.

Основной результат последнего изменения — building-macro F1/IoU при пороге,
выбранном на полном VAL под macro FPR ≤ 0.05. Также выводятся precision, recall,
фактический FPR, AP после существующего квантования scores и число доступных
зданий. Для F1/IoU сохраняется cohort текущего memory-evaluator: здания с
положительным GT. Здания без положительных меток сохраняются в FPR. Неопределимая
метрика — `NA`, а не искусственный ноль.

## Данные и причинность

Повторно используются canonical benchmark, frozen stack, predicted semantic
probabilities и spatial DINOv2 cache. Точные пути, revisions, формы, channel map,
checkpoint exposure, splits и hashes проверяются по источникам, а не по старым
числам в документации. Нормализация target features и распределение ошибок
copy-last оцениваются только на TRAIN и фиксируются до VAL/TEST.

Проверенный существующий DINO-кэш содержит patch tokens `(256, 384)`, то есть
сетку `(384, 16, 16)` для RGB-кропа 224×224. Encoder заморожен. Каждый кадр
кодируется отдельно; общая координатная система задаётся earliest anchor,
а сетка кропов — его origin `(0, 0)`. `render_observation` сохраняет эту сетку
при расширении собственного холста наблюдения. Регистрация целевого кадра
нужна только для последующего сравнения и не изменяет уже построенный прогноз.

Вход direct detector включает текущий кадр. Вход predictor содержит только
строгий префикс прошлого и известный год целевого наблюдения. Метки, current
features, current semantic predictions и detector maps не входят в прогноз.
Тест причинности изменяет/удаляет current/future данные при фиксированном
прошлом и проверяет неизменность preprocessing и результата.

Canonical cohort выбран ретроспективно: подготовка оставляет истории минимум
с тремя различными годами и пригодным геометрическим покрытием. Это ограничение
набора для оценки, а не утверждение о заранее известной доступности будущего
кадра. Для данного прошлого прогноз вычисляется независимо от наличия цели.
Когорты `all` и `at_least_three_observations` могут совпадать; в таком случае
это одна проверка, а не два независимых подтверждения.

Physical `building_id`/facade — единица split и статистики. Существующий split
сохраняется; дополнительно проверяются view/source identity и SHA. Предсказанные
семантические вероятности отделены от GT API. GT разрешён только для TRAIN-loss
и последующей оценки. Unknown и newly visible не считаются стабильными или
повреждёнными. Геометрическая поддержка не является маской окклюзий.

Для новых сравнений фиксируется одна общая поддержка: canonical RGB support
пересекается с перенесённой на пиксели поддержкой patch tokens первого,
предыдущего и текущего наблюдений. Она одинакова для stack и всех новых
детекторов; отсутствие допустимого token не превращается в отрицательный
пример. Покрытие этого поднабора сообщается отдельно. `stack-original`
сохраняет исходную полную canonical поддержку для проверки точного совпадения
со старым evaluator; его нельзя напрямую использовать как контроль карты на
другом support. Первоначальные event IDs сохраняются, в том числе события без
достаточного числа допустимых пикселей.

Последнее изменение строится из исходных покадровых аннотаций после alignment:
это **annotation-derived temporal target**, а не независимая разметка событий.
Checkpoint exposure и неизвестная независимость старого semantic TRAIN
сохраняются в provenance, если их нельзя исключить по сохранённым данным.

## Основная матрица

Для каждой семьи ConvGRU и factorized temporal/spatial Transformer реализованы
два основных масштаба `medium`/`large`; `smoke` служит только технической проверке.
Обе семьи используют одно пространство frozen DINO features, одинаковые
доступные входы и общий decoder/head внутри задачи. Сравнение с близким числом
параметров и сравнение лучшей VAL-конфигурации каждой семьи — разные срезы.
Близкое число параметров не означает равный compute; фактические время, VRAM
и optimizer steps выводятся отдельно.

Для входа DINO 384 channels измерены следующие trainable parameter counts
(H0 имеет ту же архитектуру, что actual-next):

| Размер / задача | ConvGRU | Transformer | Разница относительно меньшей модели |
|---|---:|---:|---:|
| medium / direct | 2 030 849 | 1 832 961 | 10.80% |
| medium / actual-next, H0 | 2 129 792 | 1 931 904 | 10.24% |
| large / direct | 6 485 569 | 6 708 865 | 3.44% |
| large / actual-next, H0 | 6 633 600 | 6 856 896 | 3.37% |

| Задача | Доступный вход | TRAIN objective | Профильный VAL выбор |
|---|---|---|---|
| `direct` | Прошлое и текущий кадр | Плотная карта последнего изменения | Recall при FPR ≤ 0.05, затем меньший FPR |
| `actual-next` | Только прошлое и горизонт | Следующие пространственные признаки на всех допустимых переходах | Prediction score на всех допустимых областях |
| `h0` | Только прошлое и горизонт | Признаки на надёжно известных стабильных TRAIN-областях | Prediction score на стабильных областях |

Predictive checkpoint по профильному прогнозу сохраняется отдельно от checkpoint
по downstream detection. Выборы, размеры, losses, optimizer, budget, patience и
список сравнений фиксируются до первого TEST-forward. Дополнительные FPR-бюджеты
0.10 и 0.15 меняют VAL-порог, а не требуют повторного обучения.

Для каждого основного варианта предусмотрены `history`, обученный `pair-only`
(direct) / `last-only` (predictive), `mean` без моделирования порядка и независимо
обученный `repeat-history`. Добавляются замороженные stack и copy-last. Отдельные
абляции меняют только signed/absolute delta или реальные интервалы/ordinal time.
Все производные признаки пересчитываются после вмешательства. Изменение входа
обученного checkpoint использует те же веса и порог; это не новое обучение.

Transformer адаптирует разделение temporal/spatial processing и кодирование
времени из [TSViT](https://openaccess.thecvf.com/content/CVPR2023/html/Tarasiou_ViTs_for_SITS_Vision_Transformers_for_Satellite_Image_Time_Series_CVPR_2023_paper.html)
и [TimeSformer](https://proceedings.mlr.press/v139/bertasius21a.html) для плотных
карт. Temporal modules и decoder обучаются с нуля; они не являются предобученной
видеомоделью. [V-JEPA 2](https://arxiv.org/abs/2506.09985) — методический ориентир
прогноза представлений, а не использованный checkpoint или воспроизведённый
масштаб обучения. При frozen backbone target representation фиксирован и
отделён stop-gradient; EMA encoder не нужен.

## Неожиданность и выигрыш истории

Вероятностная голова параллельно предсказывает mean/scale пространственных
latent components. Quantizer и target normalization общие у всех моделей.
Используется вероятность quantization bin, а не непрерывная плотность:

`U = -log2 P(Q(z_t) | past, target time)`.

Единица — model bits/component; сумма по каналам даёт bits/token. Минимальный
scale фиксирован; диагностируются численная saturation и coverage. Большая
variance оплачивается proper NLL. `G = NLL(last-only) - NLL(history)` строится
для одинаковых target, quantizer и support. Положительный G означает более
качественное объяснение наблюдения, а не автоматически повреждение.

Copy-last получает TRAIN-fitted error distribution, поэтому его NLL определён.
L1/cosine и NLL оцениваются отдельно на stable/changed; при H0 не требуется
одновременно воспроизводить изменение и оставлять большой остаток на нём.
Связь с [Ballé et al.](https://arxiv.org/abs/1802.01436) — вероятностное
моделирование latent variables. Здесь нет декодируемого потока, achieved
compression ratio, RGB bpp или полного оплаченного H0/H1 codec. Существующий
RGB-компрессионный baseline сохраняется.

## Сценарии A/B/C

**A, semantic state.** Сравниваются независимые predicted probabilities,
temporal mean, confidence-aware update, обучаемое состояние и повтор последнего
наблюдения. GT отсутствует в forward/update. Проверяются принятие ремонта,
исчезновение и повторное появление повреждения. Semantic F1/IoU по классам
отделены от class-conditioned change detection и downstream последнего изменения.
TRAIN-loss обучаемой памяти использует семантическое состояние; основной
checkpoint выбирается по заранее заданному downstream VAL recall/FPR. Поэтому
поклассовые semantic metrics относятся именно к этому выбранному checkpoint;
он не объявляется отдельно оптимальным checkpoint для сегментации. Контроль
repeat сохраняет последнюю реальную пару и заменяет более старое содержание
последним прошлым наблюдением; gate остаётся обучаемым.

**B, history only during training.** Одинаковый pair detector обучается без
auxiliary loss, с feature-level regularization без истории и с temporal
supervision на стабильных TRAIN-соответствиях. На истинных изменениях равенство
представлений не навязывается. Inference signature содержит только последнюю
пару; прошлые кадры используются лишь при обучении. Feature-level perturbation
не называется RGB-изменением освещения или ракурса.

**C, progression proxy.** Последняя пара, first/current и полная история
оцениваются на одном target и общих допустимых примерах. Damage classes
разрешаются по именам в активной онтологии, а не по предполагаемым ID.
REPAIRS, TEXT_OR_IMAGES, ADVERTISEMENTS и ORNAMENT_INTACT не объявляются
автоматическим ухудшением. Пропуски/unknown и недостаточное число наблюдений
не дают отрицательную метку. Target является annotation-derived **proxy**.
Его F1 не сравнивается в одном рейтинге с F1 последнего изменения. При
недостаточном покрытии сохраняются статистика и review manifest, статус
`INSUFFICIENT_DATA`; синтетика не заменяет реальные события.

Правило `2026-11-08-appearance-persistence-v1` фиксируется в resolved matrix
до обучения:

- Нужно не менее **трёх** наблюдений в строго возрастающие известные годы.
  Для оцениваемого пикселя поддержка и состояние нужных каналов должны быть
  известны во **всех** наблюдениях; пропуск оставляет target unknown.
- Any-damage объединяет `WATER_STAIN`, `MISSING_ELEMENT`, `DELAMINATION`,
  `SPALLING`, `CRACK`, `EFFLORESCENCE`, `CORROSION`. Их ID и порядок каналов
  берутся из проверенной ontology.
- Положительный target: после наблюдавшегося отсутствия any-damage начался
  новый эпизод, который присутствует минимум в **двух последних подряд**
  наблюдениях. Порог — число подтверждений, а не выдуманная точная длительность
  между годами; реальные интервалы остаются входом модели и отчётной статистикой.
- Неизменное наличие или отсутствие повреждения — stable target 0. Наблюдавшееся
  повреждение, исчезнувшее к текущему кадру, — recovery target 0. Новое повреждение
  лишь на одном последнем наблюдении — unconfirmed/unknown, а не positive.
- `REPAIRS` сбрасывает прежний damage episode. Позднее повторное появление
  должно снова получить два подтверждения. Одновременные damage и REPAIRS
  трактуются как неоднозначность, без принудительной бинарной метки.
- Минимум покрытия для обучения и выбора: по **два здания** с допустимым
  proxy и минимум **одно положительное здание** в каждом из TRAIN и VAL.
  Покрытие TEST сообщается отдельно; отсутствие достаточных событий не
  превращается в подтверждение отсутствия эффекта.

Это проверка устойчивого нового эпизода, **не** измерение роста тяжести или
площади уже существующего повреждения. `2026-11-08_progression_review_manifest.json`
содержит rule, coverage, event/building IDs и причины insufficient/unconfirmed.

## Практический критерий и отчёт

Порог практической пользы задаётся до TEST: ΔF1 ≥ 0.02 **или** ΔIoU ≥ 0.015
при ΔFPR ≤ 0.005 и Δrecall ≥ −0.01. Альтернатива — относительное снижение FPR
минимум на 20% при Δrecall ≥ −0.01 и ΔF1 ≥ −0.005. Это рабочие ориентиры этого
эксперимента, а не универсальные научные стандарты. Всегда выводятся исходные
метрики, paired bootstrap 95% CI и распределение разностей по зданиям.

Контроль выбирается на VAL из соответствующих сильных вариантов без полезной
истории; точное имя сохраняется. Для predictive успеха требуются и улучшение
профильного прогноза относительно контролей, и практический downstream-выигрыш.
Меньший NLL без выигрыша в обнаружении — результат о прогнозировании.

В конце полного или частичного запуска печатается одна основная таблица при
VAL FPR-бюджете 5%. Она различает practical benefit, выигрыш ниже порога,
отсутствие выигрыша, `INSUFFICIENT_DATA`, `INCOMPLETE` и `ERROR`. Bootstrap
direction support показывается отдельно. Выполнение без ошибок не означает
подтверждение гипотезы. Подробная таблица содержит масштабы, controls, ablations,
другие бюджеты и class/group slices.

Float maps, support, event IDs, bbox и thresholds сохраняются вместе с NPZ,
метриками и хешами. Галерея строится по фиксированному общему набору и отдельному
правилу ошибок; шкалы каждого типа задаются по TRAIN/VAL. PNG не является
единственным результатом. `report-only` пересобирает отчёт без обучения и
повторного inference базовых детекторов.

## Исполнение и сохранность

Один runner выполняет preflight, короткую техническую проверку, только недостающую
совместимую подготовку, последовательное обучение, фиксацию VAL-выборов,
замороженную оценку и отчёт. Одна GPU-задача запускается за раз; чужие процессы
не останавливаются. Ресурсы измеряются локально, длительность всей матрицы до
измерений не обещается. AMP и gradient accumulation сохраняют effective batch.
OOM не уменьшает автоматически resolution, history, capacity или budget.

Используется существующий `RunStore`/append-only `events.jsonl`. Сохраняются
resolved config, code/input hashes, dataset/split/ontology, seed и environment.
`2026-11-08_training_contract.json` дополнительно фиксирует размеры архитектур,
optimizer, losses и pretrained/trainable части до начала обучения.
Checkpoint завершённой эпохи содержит model, optimizer, scheduler и RNG.
Resume требует совместимого config/source и продолжает незавершённую стадию;
изменение научной конфигурации требует нового run. Завершённые артефакты не
перезаписываются. Ошибка отдельной модели получает явный статус; нарушение
split/ontology/causality блокирует затронутую оценку.

Отчёт собирает гистограммы по одному методу/split, сохраняет временный дисковый
буфер и читает его через readonly mmap. Полная матрица гистограмм не удерживается
в RAM; собственные временные файлы удаляются после отчёта, исходные карты остаются.

## Проверки

Фокусные тесты покрывают spatial shapes обеих архитектур, padding/support,
каузальное preprocessing, отсутствие GT в model inputs, evaluator parity,
вмешательства с сохранённой последней парой и пересчётом delta, quantized NLL/G,
repair/reappearance A, pair-only inference B, progression fixtures C,
resume/report-only и неполные/неопределимые результаты.

Обязательная проверка репозитория:

```bash
python -s -B -m unittest discover -s tests -v
```

CLI проверен фактическим `python -m data_analysis history-scenarios --help`.
Фокусные проверки моделей и A/B/C выполняются с CPU Torch; обязательная общая
проверка и сквозной smoke запускаются перед сдачей ветки. Данные и checkpoints
владельца в Cloud отсутствуют; это не заменяется результатами синтетики.

Фактические результаты CPU-проверок, сквозного запуска и проверки сохранности
описаны в [отчёте проверки реализации](2026-11-08-history-three-scenarios-validation.md).

## Команды владельцу

Получить отдельную ветку, не сливая её в базовую:

```bash
git -C /home/sasha/Facade_Change_Benchmark fetch origin codex/2026-11-08-history-three-scenarios
git -C /home/sasha/Facade_Change_Benchmark switch --track origin/codex/2026-11-08-history-three-scenarios
git -C /home/sasha/Facade_Change_Benchmark pull --ff-only origin codex/2026-11-08-history-three-scenarios
```

Если эта ветка уже существует локально, вместо `switch --track` используется
`git -C /home/sasha/Facade_Change_Benchmark switch codex/2026-11-08-history-three-scenarios`.
Чужие незакоммиченные изменения перед переключением должны оставаться сохранены;
runner не выполняет checkout, clean, reset или merge.

Во всех командах ниже директории входов заданы абсолютными путями в
`configs/2026-11-08_history_three_scenarios.json`. `conda` вызывается как обычная
программа из PATH, без shell aliases и заранее заданных глобальных переменных.
CLI использует сохранённый DINO worker Python, если исходный benchmark задаёт
другой интерпретатор. Рабочее CUDA-окружение не переустанавливается.

Preflight проверяет реальные inputs и измеряет forward/backward shapes:

```bash
cd /home/sasha/Facade_Change_Benchmark
conda run --no-capture-output -n scd_bench python -s -u -m data_analysis history-scenarios preflight \
  --config /home/sasha/Facade_Change_Benchmark/configs/2026-11-08_history_three_scenarios.json \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-08-history-three-scenarios-preflight-001
```

Отдельный CPU smoke выполняет синтетическую матрицу тем же путём обучения,
сохранения и отчёта. Он не использует реальные данные и не подтверждает гипотезу:

```bash
cd /home/sasha/Facade_Change_Benchmark
conda run --no-capture-output -n scd_bench python -s -u -m data_analysis history-scenarios smoke \
  --config /home/sasha/Facade_Change_Benchmark/configs/2026-11-08_history_three_scenarios.json \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-08-history-three-scenarios-smoke-001 \
  --device cpu
```

Одна основная команда запускает всю зафиксированную матрицу, controls и A/B/C
с seed 13; ручная цепочка отдельных моделей не требуется:

```bash
cd /home/sasha/Facade_Change_Benchmark
conda run --no-capture-output -n scd_bench python -s -u -m data_analysis history-scenarios full \
  --config /home/sasha/Facade_Change_Benchmark/configs/2026-11-08_history_three_scenarios.json \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-08-history-three-scenarios-full-001
```

Возобновление того же run с неизменной научной конфигурацией:

```bash
cd /home/sasha/Facade_Change_Benchmark
conda run --no-capture-output -n scd_bench python -s -u -m data_analysis history-scenarios resume \
  --config /home/sasha/Facade_Change_Benchmark/configs/2026-11-08_history_three_scenarios.json \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-08-history-three-scenarios-full-001
```

Пересборка отчёта из сохранённых артефактов:

```bash
cd /home/sasha/Facade_Change_Benchmark
conda run --no-capture-output -n scd_bench python -s -u -m data_analysis history-scenarios report-only \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-08-history-three-scenarios-full-001
```

Новый научный config требует нового output, например суффикса `-002`; это не
повод перезаписывать `-001`. Повторный smoke также получает новый output, либо
явный `resume` исходного smoke с тем же `--device cpu`.

## Основные выходы

Для команды full выше главная таблица находится в:

`/home/sasha/Facade_Change_Benchmark/runs/2026-11-08-history-three-scenarios-full-001/2026-11-08_main_table.csv`.

В том же каталоге сохраняются:

- `2026-11-08_main_table.json`, `2026-11-08_report.txt`, `2026-11-08_report.json`;
- `2026-11-08_detailed_table.csv`, `2026-11-08_metrics.csv`,
  `2026-11-08_per_building.csv`, `2026-11-08_comparisons.csv`,
  `2026-11-08_prediction_metrics.csv`, `2026-11-08_thresholds.json`;
- `2026-11-08_resolved_config.json`, `2026-11-08_resolved_matrix.json`,
  `2026-11-08_data_contract.json`, `2026-11-08_feature_normalization.json`;
- `2026-11-08_hardware.json`, `2026-11-08_profile.json`,
  `2026-11-08_storage_budget.json`, `2026-11-08_previous_evidence.json`,
  `2026-11-08_training_records.json`, `2026-11-08_progression_review_manifest.json`;
- `2026-11-08_maps_manifest.json`, `2026-11-08_frozen_val_decisions.json`,
  float NPZ в `maps/` и их атомарные hash-маркеры в `units/`;
- `2026-11-08_gallery/2026-11-08_index.html`, галереи фиксированных примеров
  и ошибок, `2026-11-08_gallery/2026-11-08_manifest.json` со шкалами и правилами;
- `models/<model-id>/2026-11-08_epoch-<number>.pt`,
  `2026-11-08_latest.json`, `2026-11-08_selection.json` и `2026-11-08_curves.json`;
- `run.json`, `request.json`, `experiment_contract.json`, `events.jsonl` —
  существующий формат run/ledger и цепочка происхождения.

Все epoch checkpoints сохраняют optimizer state; перед длительным запуском
нужно учитывать дисковое место наряду с VRAM/RAM. Их сохранение не ограничено
только последним checkpoint, и автоматического удаления прошлых результатов нет.
При 384-channel DINO и завершении всех 60 эпох основной матрицы только payload
`parameters + Adam first/second moment` составляет примерно **156 GiB**
(число параметров × 12 bytes × epochs). Это не полный размер run: отдельно
нужны A/B/C checkpoints, RNG/scheduler/служебные данные, float maps для двух
выборов checkpoint, RGB и галереи. Early stopping может уменьшить расход.
Preflight записывает расчёт по фактической матрице и свободное место в
`2026-11-08_storage_budget.json`; оценка не обещает верхнюю границу всего run.
