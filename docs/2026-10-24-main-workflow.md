# Основной workflow: prepare → train → benchmark → gallery

Решение от 2026-10-24: один основной интерфейс проекта с четырьмя стадиями.
Реальные данные, синтетика, PREVIOUS/EARLIEST/HISTORY и robustness остаются
явными вариантами внутри стадий. Обучение выполняется при необходимости;
для zero-shot методов и готовых checkpoint стадия train пропускается.

```bash
python -m facade_change workflow --help
python -m facade_change workflow prepare --help
python -m facade_change workflow train --help
python -m facade_change workflow benchmark --help
python -m facade_change workflow gallery --help
```

После установки проекта в текущую среду `facade workflow ...` эквивалентно
`python -m facade_change workflow ...`. Рабочий каталог — checkout репозитория.
Команда `STAGE TASK --help` показывает прежние флаги задачи. Стадии запускаются
отдельно; автоматического `run-all` в этой версии нет. Выбор входов, checkpoint,
split, методов и нового `--out` остаётся явным. Команды ничего не выбирают по
принципу «последний каталог в runs» поверх поведения существующего backend.

## Прогресс подготовки

С 2026-10-25 `tqdm` включён по умолчанию: `prepare crops` показывает проверку
сохранённых выравниваний, обработку пар и окон кропов; `synthetic` — число
сгенерированных H0/H1 cases; `history` — кадры двух синтетических историй;
`augmentations` — исходные кропы и фиксированные TRAIN-варианты; `real` —
проверку входов, просмотр видов и экспорт историй; `robustness` — реальные пары.
Видны обработанное количество, общее количество, время, скорость и оценка
остатка. Пары/окна, исключённые по прежним правилам, также считаются обработанными;
число экспортированных кропов остаётся в итоговой сводке.

Проверки входных хешей и финальное хеширование crop/synthetic/augmentation
артефактов имеют отдельные индикаторы. Полосы выводятся в stderr; формат JSON
и итоговых файлов сохраняется. Дополнительные флаги не нужны.
Обновление исходников выполняйте после окончания текущего запуска: сохранённые
входные хеши проверяются повторно. Уже работающий процесс новых полос не получит.

## Передача данных между стадиями

| Стадия | Задачи | Вход → выход |
|---|---|---|
| prepare | dataset, align, crops | COCO + RGB + подтверждённые метаданные → dataset/split; RGB + manifest → alignment; сохранённая геометрия → crop dataset |
| prepare | crop-histories, history-features | Сохранённая сессия alignment → общая сетка RGB-историй → frozen DINOv2 по датам и реальные интервалы лет |
| prepare | targets / history-targets | Существующая COCO-разметка + `--features` историй или `--crops` пар → общие правила масок и аннотационных целей; RGB/признаки сохраняются |
| prepare | history-targets | Существующие целевые COCO-маски + геометрия историй → автоматические метки различий, неизвестные области и галерея |
| prepare | history-fractions | Сохранённые поклассовые карты history-targets → доли изменений и маска доступных целей для регрессии |
| train / benchmark | history-models | Готовые DINO-признаки + проверенные или явно разрешённые автоматические метки → pair/pooled/GRU/Transformer и сравнение |
| prepare | augmentations, synthetic, history | Подготовленный semantic dataset → фиксированные кропы/аугментации или пятикадровые истории; reviewed RGB crop dataset → контролируемые H0/H1 |
| prepare | real-audit, real, robustness, transitions | Сохранённые реальные данные и геометрия → audit → подготовленные истории/варианты наблюдений/кандидаты переходов |
| train | segmentation, msdzip, arib, bcm | TRAIN и отдельный VAL → training run, checkpoint и метрики; готовые модели можно переиспользовать |
| benchmark | synthetic, real, pilot, robustness, semantic, segmentation | Подготовленный набор + явные модели/маршруты → сохранённые карты, thresholds, метрики и run.json |
| benchmark | metrics, calibration, real-lofo, final-fpr, fusion, temporal-snapshot, semantic-updates, continuous-updates | Сохранённые карты → оценки, калибровка или научно определённая композиция; разные протоколы сохраняют свои контракты |
| gallery | histories, real, real-crops, robustness, robustness-results, comparison, errors, sheets, lofo, bundle, atlas | Сохранённые входы/карты/результаты → HTML, таблицы, PNG-листы, архив историй или локальный atlas |

`dataset.json`, reviewed crop dataset и `real_temporal_prepared` — разные
существующие контракты. Нельзя передавать один в команду, ожидающую другой.
Датасет для сегментации задаётся через `--dataset`; H0/H1 benchmark принимает
`--dataset-run` от `prepare synthetic`; real benchmark принимает `--prepared`
от `prepare real`. Новая оболочка сохраняет проверки каждого формата.

## Реальные данные и семантическая модель

Имена каталогов ниже — пример новых результатов, не существующие артефакты.
Замените COCO/RGB и paths-конфигурацию своими входами. Для возобновления работы
с готовыми данными начните с соответствующей следующей команды.

```bash
python -m facade_change workflow prepare dataset /path/to/coco.json /path/to/images \
  --out runs/2026-10-24-dataset-001

python -m facade_change workflow prepare augmentations \
  --dataset runs/2026-10-24-dataset-001 \
  --out runs/2026-10-24-augmentations-001

python -m facade_change workflow train segmentation \
  --dataset runs/2026-10-24-dataset-001 \
  --augmentations runs/2026-10-24-augmentations-001 \
  --methods-config configs/methods.local.json \
  --out runs/2026-10-24-segmentation-001 --device cuda:0

python -m facade_change workflow benchmark segmentation \
  --evaluate-run runs/2026-10-24-segmentation-001 \
  --evaluation-mode originals --splits val \
  --out runs/2026-10-24-segmentation-val-001
```

Оценка сохраняет собственную галерею целых фотографий. TRAIN/VAL/TEST не
объединяются для подбора порогов; TEST включается только явно. Подробности
артефактов и локальных моделей: [семантическая ветвь](2026-10-11_dino_semantic_training.md).

Обучение не меняет чужие `methods.local.json`, semantic config или подготовленный
real snapshot. Чтобы оценить новый checkpoint в change benchmark, явно создайте
его конфигурацию/подготовленный набор по существующему контракту. Использование
старого frozen snapshot означает использование закреплённых в нём моделей.

## RGB, синтетика и кодеки

Путь RGB сохраняется: исходные фотографии → выравнивание → RGB-кропы → H0/H1.
Семантические маски не подменяют исходный сигнал.

```bash
python -m facade_change workflow prepare align \
  --manifest /path/to/prepared/manifest.json --methods sift --split all --limit 0 \
  --out runs/2026-10-24-alignment-001

python -m facade_change workflow prepare crops \
  --batch-run runs/2026-10-24-alignment-001 --methods sift \
  --out runs/2026-10-24-crops-001

python -m facade_change workflow prepare synthetic \
  --crop-run runs/2026-10-24-crops-001 \
  --config configs/2026-10-04_h0h1_controls.json \
  --out runs/2026-10-24-synthetic-001

python -m facade_change workflow benchmark synthetic \
  --dataset-run runs/2026-10-24-synthetic-001 --methods rgb_diff ssim \
  --out runs/2026-10-24-benchmark-001
```

Для RGB-DIFF/SSIM обучение не требуется. Этот benchmark сразу создаёт
`gallery.html`, `metrics.json`, `summary.json`, `threshold_selection.json`,
карты и маски: стадия gallery здесь уже выполнена renderer-ом benchmark.
Кодеки обучаются через `workflow train msdzip/arib/bcm`; флаги и допустимый
TRAIN H0 input определяются их `--help` и [codec contract](2026-10-04_residual_compression.md).

`gallery comparison --run ... --out ...` — существующий анализ фиксированного
набора из шести методов, включая MSDZip и GeoSCD. Он не подходит для сокращённого
RGB-DIFF/SSIM запуска. Его [контракт](../data_analysis/README.md) сохранён.

## Истории и robustness

- `prepare crop-histories` → `prepare history-features` собирает новые
  [канонические истории RGB-кропов и frozen DINOv2](2026-10-25_history_features.md)
  с кэшем, альтернативными снимками одного года и явными пропусками дат.
- `prepare history-targets` переносит [целевые маски в эти истории](2026-10-25_history_targets.md)
  без повторного DINO. `train history-models` обучает модели по пригодным меткам;
  неразмеченное не становится отрицательным классом.
- `prepare history` строит синтетические истории из semantic dataset.
- `prepare transitions` использует готовое выравнивание; `gallery histories`
  объединяет сохранённые synthetic/real истории для просмотра.
- `prepare real-audit` → `prepare real` → `benchmark pilot` → `benchmark real`
  переиспользует существующие реальные хронологии. Флаги `--run`, `--snapshot`,
  `--audit` и `--prepared` описаны в [real workflow](2026-10-17_real_temporal_night.md).
- `benchmark real --modes previous earliest history ...` сохраняет прежнюю
  причинную историю и различия целевых меток. Это запуск готовых моделей без обучения.
- `benchmark real-lofo --run ... --out ...` запускает перенесённую из LOFO-ветки
  калибровку; `gallery real --run ... --out ...` пересобирает отчёт из карт.
- `prepare robustness` → `gallery robustness` → сохранённый ручной выбор →
  `benchmark robustness` → `gallery robustness-results` сохраняет обязательный
  review и отдельный [robustness protocol](2026-10-17_real_robustness.md).

Существующий `python -m facade_change temporal-experiment` также остаётся:
это составной TRAIN/VAL опыт, включающий обучение MSDZip, синтетические и реальные
истории. Он явно описан [отдельно](2026-10-14_temporal_experiment.md), поскольку его
вызов может обучать модели. Его report-only/resume не заменены новой логикой.

## Atlas

```bash
python -m facade_change workflow gallery bundle \
  --prepared /path/to/real-prepared --include-originals \
  --out /path/to/2026-10-24-atlas-input.tar.gz

python -m facade_change workflow gallery atlas \
  --bundle /path/to/2026-10-24-atlas-input.tar.gz \
  --out runs/2026-10-24-atlas-001 --prepare-only

python -m facade_change workflow gallery atlas --serve runs/2026-10-24-atlas-001
```

Сервер доступен только на localhost. Atlas использует сохранённые наблюдения,
геометрию и GT; экспорт не запускает обучение или inference. Требования к
подготовленным кропам и bundle сохранены в [atlas contract](2026-10-18-facade-atlas.md).

## Локальный скрипт статистики и выравнивания

По уточнению владельца от 2026-10-24 `data/dataset_statistics.py` исключён из
этого PR. Его новая локальная версия имеет режимы `--mode stats` и `--mode prepare`:
статистика зданий/видов и инкрементальное попарное выравнивание SIFT/LoFTR/GeoSCD
с проверкой cache, планом запуска и галереей для review. Она вызывает существующие
`facade_change batch` и `facade_change geoscd`; эти команды сохраняются.

Файл остаётся под локальным управлением владельца. В этом PR нет ни его копии
в пакете, ни launcher-а, ни команды `prepare statistics`, ни связанных тестов
старой версии. Правило игнорирования `data/` остаётся исходным. Новая локальная
версия просмотрена как исходный код; её GPU-работоспособность здесь не проверялась.

Подключение от 2026-10-24 читает **завершённую сессию** этого скрипта через
`--alignment-session`. `data/dataset_statistics.py` по-прежнему не копируется
в пакет и не меняется. Повторный поиск соответствий и загрузка моделей для
экспорта не нужны. Дождитесь окончания текущего расчёта и quality worker, затем
обновите рабочую ветку: не меняйте исходники посреди запущенных subprocess.

```bash
cd /home/sasha/Facade_Change_Benchmark
git pull --ff-only

conda run --no-capture-output -n lposs310 python -B -m facade_change workflow prepare crops \
  --alignment-session data/aligned \
  --tile-size 256 --stride 128 --min-valid-fraction 0.8 \
  --out data/crops
```

`data/aligned` разрешается только при наличии ровно одной сессии. При нескольких
передайте конкретный путь из строки `Reports:`, например
`data/aligned/sessions/report-2`; выбор «самой свежей» не выполняется.
Без дополнительного подтверждения незавершённая/failed сессия отклоняется;
plan-only отклоняется всегда. Новый каталог `data/crops`
должен отсутствовать. Если рядом с manifest нет `split.json`, добавьте
`--split-file /path/to/existing/prepared/split.json`: принадлежность здания
должна совпадать с manifest; новый split не создаётся.

Если preparation упал во время quality worker, отдельный `--mode evaluate`
завершает свой quality-отчёт, но **не завершает родительский `run.json`**.
Для этого случая передайте точный завершённый отчёт явно:

```bash
python -B -m facade_change workflow prepare crops \
  --alignment-session data/aligned/sessions/report-1 \
  --quality-report data/aligned/sessions/report-1/quality/report-2 \
  --out data/crops
```

Можно указать и напечатанный путь к `quality_report.html`. Импорт проверяет
завершение quality run, его хеши `pairs.json`/`inputs.json`, качество обработки
каждой выбранной пары и сохранённые хеши её run/геометрии. Недостаток Canny-границ
допустим; ошибка обработки или отсутствие provenance выбранной пары — отказ.
Исходные решения review сопоставляются с проверенными pairs и run-хешами quality.
Существующий `.preparation.lock` должен быть свободен: активный prepare не
импортируется. Оставшийся после остановки статус `running` допустим только при
этих же проверках. Статус исходной сессии и её файлы **не переписываются**;
новый экспорт сохраняет `alignment_session_completion` с исходным статусом,
основанием `explicit_quality_snapshot` и хешем выбранного quality run.
Самый новый отчёт автоматически не выбирается. Повторный alignment не нужен.

Результат: `run.json`, `summary.json`, `split.json`, индекс JSON/CSV с путём в
`summary.json`, `excluded_pairs.csv`, RGB-кропы, маски геометрической поддержки,
`gallery.html`. Он принимается прежней командой синтетики без конвертации:

```bash
python -m facade_change workflow prepare synthetic \
  --crop-run data/crops --config configs/2026-10-04_h0h1_controls.json \
  --out data/synthetic
python -m facade_change workflow benchmark synthetic \
  --dataset-run data/synthetic --methods rgb_diff ssim \
  --out data/synthetic_benchmark
```

Для **реальных историй** подключите ту же сессию к существующему аудиту.
`--run` и `--snapshot` — сохранённый temporal experiment и его соответствующий
снимок калибровки; пути ниже обозначают входы, их нужно заменить своими.
`--dataset` — текущий semantic dataset.json с COCO и reviewed split.
Это не файл `pairs.json` и не результат экспорта RGB-кропов.

```bash
python -m facade_change workflow prepare real-audit \
  --run /path/to/existing/temporal-experiment \
  --snapshot /path/to/its/temporal-snapshot \
  --dataset /path/to/current/dataset.json \
  --alignment-session data/aligned \
  --splits train val --out data/history_audit
python -m facade_change workflow prepare real \
  --audit data/history_audit --out data/real_histories
```

Полученный `data/real_histories/manifest.json` имеет прежний контракт
`real_temporal_prepared`: пригоден для `benchmark real --prepared data/real_histories`,
PREVIOUS/EARLIEST/HISTORY, real gallery и atlas bundle. Модели и пороги
берутся из указанного frozen experiment/snapshot; переход к новым весам или
перекалибровка не подразумевается. Существующий отбор real-кропов по разметке
сохраняется и явно остаётся GT-enriched diagnostic subset. TEST включается
только явно в `--splits`, без подбора порогов. Стадия `prepare real` уже создаёт
`gallery.html`. Atlas по-прежнему требует 512×512 и одну историю на здание.

Неразмеченный research loader также принимает путь сессии единственным элементом
`alignment_runs` в temporal config; он выбирает кропы по геометрической поддержке,
не загружает semantic GT и не меняет синтетическую ветвь.
Для отдельного quality rerun `prepare real-audit` принимает тот же
`--quality-report`; research config использует `alignment_quality_report`.

Правила подключения:

- `selected_method` и `selection_status` сохраняются. `--methods` для такого
  crop export не используется: нельзя незаметно заменить выбранный метод.
  Неопределённые пары попадают в исключения. Нельзя смешивать сессию со списком
  старых alignment-запусков или автопоиском для подстановки альтернатив.
- GeoSCD переносится как **плотное поле**, без подмены гомографией. Нужен `accept`
  в hash-bound review; SIFT/LoFTR с пройденным gate сохраняют статус provisional.
  Явный `accept` может принять отклонённую gate гомографию, но исходный статус
  остаётся в provenance. Для изменения решения используйте копию `review.csv`
  и новый вызов исходного скрипта с `--reuse-existing --review-csv ...`.
  Не редактируйте завершённую сессию на месте.
- Сохраняются ссылки и хеши Canny/cycle/drift-отчёта. Его рейтинг рекомендательный
  и **не меняет** метод автоматически; это не независимая геометрическая GT.
- Все межгодовые пары доступны для парных RGB-кропов. Истории используют только
  прямые преобразования к самому раннему наблюдению того же вида. Отсутствующие
  даты отмечаются; выравнивание через цепочку не подставляется. Несколько кадров
  одного года остаются прежним блокером `ambiguous_same_year_observations`.
- Homography-кропы сохраняют native union canvas; GeoSCD — native reference grid.
  RGB интерполируется из оригиналов один раз на кроп, alpha и границы входят в
  поддержку. Предсказанная окклюзия GeoSCD не заменяет геометрическую поддержку.
  Истории обеих геометрий лежат на native earliest-reference grid.
- Пары проверяются по SHA-256, зданию, виду, годам, размеру и split. При допустимом
  cache reuse старые пути/COCO ID в исходном run не переписываются; используются
  проверенные текущие наблюдения из manifest. Session, review, geometry и dense
  field имеют проверяемые хеши. Изменённые входы отклоняются.

Это соединение стадий, а не автоматический запуск обучения и всех benchmark.
`prepare dataset`, `synthetic`, `history`, `real` и `robustness` сохраняют прежние
форматы и научные протоколы.

## Локальные отчёт и галереи

```bash
python -m facade_change workflow benchmark temporal-snapshot \
  --run /path/to/temporal-experiment \
  --out runs/2026-10-24-temporal-snapshot-001

python -m facade_change workflow gallery sheets \
  --report /path/to/semantic-update/report.html --split train \
  --out runs/2026-10-24-sheets-001

python -m facade_change workflow gallery lofo \
  --run /path/to/night-run --calibration-run /path/to/completed-real-val-lofo \
  --out runs/2026-10-24-lofo-gallery-001
```

- `temporal-snapshot` принимает составной temporal experiment, включая активный
  запуск. Фиксирует список записанных units, оценивает только полные synthetic VAL
  группы, калибрует scalar-пороги по VAL H0 и сравнивает PREVIOUS/HISTORY. Неполные
  группы остаются в coverage без метрик. TEST/real и inference не выполняются.
  Прежний префикс выходов `2026-10-17_quick_*` сохранён для совместимости.
- `sheets` принимает HTML semantic-updates/directed-delta с соседними изображениями
  или review-архив через `--archive`. Компоновка не меняет сохранённые маски/пороги.
  Разные RGB одного года в одной sequence запрещено склеивать в одну хронологию.
- `lofo` принимает законченный `night_real_val_lofo_completed`, восстанавливает
  сохранённые решения пяти прежних методов и сверяет support hash и TP/FP/FN/TN.
  Нового подбора порогов нет; AnyChange сохраняет native mask. Это только real VAL.

Выход каждой команды должен быть новым. Все прежние имена локальных скриптов
сохранены как короткие launcher-ы; новые инструкции используют workflow.

## Что объединено и что ещё переносить

Объединены изменения checkpoint/provenance, LOFO-калибровки и final-FPR из
`b93ec1cbdfe0c8f0e2cb38f10262a9d918b99980` и atlas из
`4f1c71451788a587716ea2dbc1b107686469aae9` поверх semantic-update
`a65f75a916ab0ad25e1ec3669520812e221a0884`. Общие предки учтены при трёхстороннем
слиянии; научные алгоритмы этих веток не переписаны.

| Код | Решение в этом изменении |
|---|---|
| `scripts/prepare_dataset.py` | Parser принадлежит `facade_change.dataset_setup`; прежний файл — тонкий совместимый вход |
| Три Python-скрипта atlas/collector/export | Реализация в `facade_change/gallery/`; старые имена делегируют в пакет |
| Реальные runner-ы, metrics, LOFO, fusion | Доступны из стадий workflow; сохранены форматы, пороги, resume и CLI |
| Датированные semantic update/error audit | Пока сохранены на прежних путях: frozen runs сверяют source hashes |
| `build_facade_gallery.py` | Реализация в `facade_change/gallery/report_sheets.py`, команда `gallery sheets` |
| `make_old_methods_gallery.py` | Реализация в `facade_change/gallery/lofo.py`, команда `gallery lofo`; одинаковые функции масок/метрик переиспользуются из владельцев |
| `scripts/2026-10-17_temporal_quick_report.py` | Реализация в `facade_change/research/temporal_quick_report.py`, команда `benchmark temporal-snapshot` |
| `data/dataset_statistics.py` | Исключён из PR; новая локальная версия владельца сохраняется без замены |
| Корневой `2026-10-18-collect-prepared.py` | Та же реализация, что уже перенесена в `gallery/history_bundle.py`; добавлен только совместимый launcher |
| Тесты | 11 dated-файлов переименованы по функции; убран ручной loader из test_scorers; проверки доступны обычному discovery |

Локальные изменения получены в `2026-10-24-191231-942656-facade-local-code.tar.gz`
(SHA-256 `092e095427d408f6f0b9165496ba3c9403b9ac46110b993f5b79563bd3f88751`,
исходный HEAD `a65f75a916ab0ad25e1ec3669520812e221a0884`). Все семь изменённых
tracked-файлов semantic adapters/config/routes, их тестов и инструкции включены
побайтно как в архиве: строгая SAM3-загрузка с проверкой неактивной ветви и явное
разрешение доверенной LPOSS-загрузки. Проверки cache/source hashes сохранены.

`semantic-checkpoint-loading-fix.patch` содержит ровно те же семь изменений,
что `changes.patch` (различаются только длины Git hash в заголовках); дополнительная
копия patch в корне не нужна. `HEAD.txt` и `changes.patch` — сведения о передаче,
их содержимое учтено здесь и в Git, а не как новый рабочий код.

Старая статистика из архива совпадала с файлом коммита
`b5bdb7b8680974a563bd871d01d813177e863ab1`, но её перенос, инструкция и 11 тестов
исключены из итогового PR по просьбе владельца: отдельная локальная версия уже
содержит более новую подготовку выравниваний.
Два локальных документа `2026-10-11_frozen_geoscd_fusion.md` и
`2026-10-11_method_complementarity.md` сохранены с пометкой об исторических командах
и ссылкой на действующий `data_analysis`.

Оставшийся перенос кода:

1. Переносить `scripts/real_eval` в пакет по ответственности, сохраняя совместимые
   входы и численный паритет. Общие функции извлекать после проверки потребителей.
2. Выносить выбор иллюстраций и параметры отдельных опытов в конфигурации.
   PREVIOUS/EARLIEST/HISTORY, LOFO и разные варианты semantic updates остаются
   самостоятельными научными протоколами.
3. Удалять старые домашние копии и совместимые launcher-ы только после переключения
   документированных команд, локальных вызовов и тестов. Старый dataset audit
   пока импортирует dated preparation script: эта связка ещё нужна.

Source hashes старых frozen evaluations не ослабляются. Для точного повторения
старого error audit используйте зафиксированную в его provenance версию кода
(и соответствующие dirty changes, если они были). Слияние LOFO уже меняет некоторые
зависимости, например `night_report.py`; совпадение алгоритма не означает совпадение
хеша исходника. Исходные commit остаются доступны в Git.

## Проверка

```bash
python -B -m unittest discover -s tests -p 'test_workflow.py' -v
python -B -m unittest discover -s tests -p 'test_alignment_session.py' -v
python -B -m unittest discover -s tests -v
```

Workflow-тест сравнивает новые и прежние H0/H1 команды на существующем небольшом
RGB fixture: выбор порогов, TP/FP/FN/TN и хеши score/prediction должны совпадать;
исходные кропы должны остаться неизменными. CLI-проверки не запускают GPU и
не скачивают веса. Проверки настоящего обучения требуют существующих зависимостей
Torch/pycocotools; их отсутствие не подтверждает работоспособность обучения.

`test_saved_galleries.py` проверяет одинаковые PNG из HTML/архива, отказ при
конфликте хронологий и совпадение масок сохранённой LOFO-калибровки без изменения
входов. `test_temporal_quick_report.py` проверяет полное/неполное покрытие,
пороговые метрики, PREVIOUS/HISTORY и отказ при повреждённом cache.

`test_alignment_session.py` проверяет смешанные homography/dense сессии, точные
RGB и alpha, неизменность входов, отказ при изменении hash/review/split,
запрет переопределения выбора и незавершённых сессий. Сквозные fixtures проходят
через синтетический benchmark с настоящим RGB-DIFF и через real preparation,
PREVIOUS/EARLIEST/HISTORY с подменой только модельного адаптера, bundle и atlas
export. Это CPU-контрактные проверки, не измерение качества настоящих моделей.
