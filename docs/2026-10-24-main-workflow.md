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

## Передача данных между стадиями

| Стадия | Задачи | Вход → выход |
|---|---|---|
| prepare | dataset, align, crops | COCO + RGB + подтверждённые метаданные → dataset/split; RGB + manifest → alignment; сохранённая геометрия → crop dataset |
| prepare | augmentations, synthetic, history | Подготовленный semantic dataset → фиксированные кропы/аугментации или пятикадровые истории; reviewed RGB crop dataset → контролируемые H0/H1 |
| prepare | real-audit, real, robustness, transitions | Сохранённые реальные данные и геометрия → audit → подготовленные истории/варианты наблюдений/кандидаты переходов |
| train | segmentation, msdzip, arib, bcm | TRAIN и отдельный VAL → training run, checkpoint и метрики; готовые модели можно переиспользовать |
| benchmark | synthetic, real, pilot, robustness, semantic, segmentation | Подготовленный набор + явные модели/маршруты → сохранённые карты, thresholds, метрики и run.json |
| benchmark | metrics, calibration, real-lofo, final-fpr, fusion, semantic-updates, continuous-updates | Сохранённые карты → оценки, калибровка или научно определённая композиция; разные протоколы сохраняют свои контракты |
| gallery | histories, real, real-crops, robustness, robustness-results, comparison, errors, bundle, atlas | Сохранённые входы/карты/результаты → HTML, таблицы, архив историй или локальный atlas |

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
| Тесты | 11 dated-файлов переименованы по функции; убран ручной loader из test_scorers; проверки доступны обычному discovery |

Следующий перенос кода выполняется после включения локальных изменений владельца:

1. Сохранить семь изменённых tracked-файлов semantic routes и уникальные локальные
   gallery/quick-report/collector scripts. В этой ветке семь файлов оставлены
   побайтно равными исходному Git-коммиту; содержимое локальных правок сюда не попадало.
2. Переносить `scripts/real_eval` в пакет по ответственности, сохраняя совместимые
   входы и численный паритет. Общие функции извлекать после проверки потребителей.
3. Выносить выбор иллюстраций и параметры отдельных опытов в конфигурации.
   PREVIOUS/EARLIEST/HISTORY, LOFO и разные варианты semantic updates остаются
   самостоятельными научными протоколами.
4. Удалять старые домашние копии и совместимые launcher-ы только после переключения
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
python -B -m unittest discover -s tests -v
```

Workflow-тест сравнивает новые и прежние H0/H1 команды на существующем небольшом
RGB fixture: выбор порогов, TP/FP/FN/TN и хеши score/prediction должны совпадать;
исходные кропы должны остаться неизменными. CLI-проверки не запускают GPU и
не скачивают веса. Проверки настоящего обучения требуют существующих зависимостей
Torch/pycocotools; их отсутствие не подтверждает работоспособность обучения.
