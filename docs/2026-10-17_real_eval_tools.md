# Перенос локальных инструментов анализа в репозиторий

Скрипты из `/home/sasha/` перенесены в `scripts/real_eval/`. Аргументы прежних
команд сохранены. Старые файлы и результаты не удаляются. Запускать скрипты можно
из любого каталога: собственный checkout определяется относительно файла;
`--repo` позволяет явно выбрать его для восстановления GT и GeoSCD.

| Старый файл | Файл в репозитории | Назначение |
|---|---|---|
| `real_temporal_gallery.py` | `scripts/real_eval/real_temporal_gallery.py` | Просмотр сохранённых реальных историй, доступных методов и дополнений |
| `geoscd_real_retry.py` | `scripts/real_eval/geoscd_real_retry.py` | Отдельный возобновляемый GeoSCD, повторное использование готовых пар |
| `real_gt_metrics.py` | `scripts/real_eval/real_gt_metrics.py` | Восстановление COCO GT на прежней геометрии и метрики сохранённых карт |
| `real_gt_calibration.py` | `scripts/real_eval/real_gt_calibration.py` | LOFO-калибровка, сочетания и сравнение истории на общих парах/пикселях |
| `real_gt_slices.py` | `scripts/real_eval/real_gt_slices.py` | Классовые срезы; вызывается автоматически из калибровки |
| `temporal_tables.py` | `scripts/real_eval/temporal_tables.py` | Таблицы из существующего CSV |
| `temporal_fusion.py` | `scripts/real_eval/temporal_fusion.py` | Сочетания сохранённых синтетических VAL-карт |

Генератор quick-report этим переносом не добавляется. Существующий snapshot
остаётся входом. Скрипты не ищут последнюю папку автоматически и не меняют
исходные запуски. Для каждого нового отчёта нужен новый каталог.

## Реальные метрики после завершения GeoSCD

Следующие пути — текущие запуски владельца. Первая команда считает метрики по
всем реальным VAL-историям и переиспользует отдельные результаты GeoSCD. Вторая
подбирает итоговые пороги на остальных фасадах (LOFO), оценивает сочетания на
общем покрытии и автоматически строит классовые срезы. Модели не запускаются.

```bash
cd /home/sasha/Facade_Change_Benchmark

/home/sasha/miniconda3/envs/scd_bench/bin/python -u scripts/real_eval/real_gt_metrics.py \
  --run runs/2026-10-14-temporal-experiment-002 \
  --snapshot runs/2026-10-17-temporal-val-snapshot-001 \
  --review runs/2026-10-17-real-review-002/summary.json \
  --geoscd-run runs/2026-10-17-geoscd-real-all-001 \
  --all-real-val \
  --out runs/2026-10-17-real-gt-all-003

/home/sasha/miniconda3/envs/scd_bench/bin/python -u scripts/real_eval/real_gt_calibration.py \
  --metrics-run runs/2026-10-17-real-gt-all-003 \
  --out runs/2026-10-17-real-calibration-all-003
```

Для обсуждения результатов нужны `summary.txt`, `comparisons.csv`,
`class_macro.csv`, `coverage.csv` и `calibration_audit.csv` из второго каталога.
Первый также содержит `gt_gallery.html`, `per_pair.csv`, `aggregate.csv`,
`history_comparison.csv`, GT-артефакты и provenance.

GT здесь — изменение набора аннотированных COCO-классов на общей известной
поддержке. Неразмеченные пиксели остаются неизвестными. Это не независимая
экспертная разметка всех визуальных изменений. Классовый срез характеризует
локализацию изменений в области класса, а не предсказание класса детектором.
Ремонт тоже является направленным изменением. Проверка неба сюда не добавлена.

P/R/F1/IoU усредняются по положительным кадрам внутри фасада, затем по фасадам;
FPR использует известные отрицательные пиксели. CSV сохраняют доли, текст —
проценты. Суммы TP/FP/FN/TN — отдельные счётчики, по ним нельзя восстановить
макрометрику. LOFO относится к итоговому порогу: прежние веса, история и порог
обновления памяти не переобучаются. Это exploratory VAL, не независимый TEST.

## Обновить небольшую галерею

```bash
/home/sasha/miniconda3/envs/scd_bench/bin/python -u scripts/real_eval/real_temporal_gallery.py \
  --run runs/2026-10-14-temporal-experiment-002 \
  --snapshot runs/2026-10-17-temporal-val-snapshot-001 \
  --geoscd-run runs/2026-10-17-geoscd-real-all-001 \
  --max-facades 6 --crops-per-facade 2 \
  --out runs/2026-10-17-real-review-all-003
```

Файл `gallery.html` содержит сохранённые результаты; `coverage.csv` показывает
пропуски. Эта галерея не является интерфейсом утверждения новых аугментаций.

## Отдельный GeoSCD

Пример **нового** запуска всех реальных VAL-пар обоих режимов:

```bash
/home/sasha/miniconda3/envs/scd_bench/bin/python -u scripts/real_eval/geoscd_real_retry.py \
  --run runs/2026-10-14-temporal-experiment-002 \
  --snapshot runs/2026-10-17-temporal-val-snapshot-001 \
  --review runs/2026-10-17-real-review-002/summary.json \
  --all-real-val \
  --out runs/2026-10-17-geoscd-real-all-002
```

Не запускайте этот пример параллельно с текущим GeoSCD без отдельной необходимости.
Готовый запуск продолжается теми же аргументами и `--resume`;
`--retry-failed` дополнительно повторяет отказы. Текущий работающий процесс
завершает вычисления своим загруженным кодом; перенос скриптов его не перезапускает.

Поскольку runner фиксирует собственный SHA-256, перенос известного старого
скрипта в репозиторий требует **явного** `--resume --upgrade-runner`.
Разрешены только два проверенных прежних SHA-256; входы, настройки, исходный
код GeoSCD и выбранные пары должны совпасть. Сохраняется запись об обновлении.
Это не обход проверки неизвестного изменения кода. Идентичная репозиторная
версия далее продолжается обычным `--resume`.

## Синтетические таблицы и сочетания

```bash
/home/sasha/miniconda3/envs/scd_bench/bin/python scripts/real_eval/temporal_tables.py \
  runs/2026-10-17-temporal-val-snapshot-001/2026-10-17_quick_units.csv \
  --out runs/2026-10-17-full-tables-003

/home/sasha/miniconda3/envs/scd_bench/bin/python scripts/real_eval/temporal_fusion.py \
  --run runs/2026-10-14-temporal-experiment-002 \
  --units runs/2026-10-17-temporal-val-snapshot-001/2026-10-17_quick_units.csv \
  --budgets 0.01 0.05 \
  --out runs/2026-10-17-synthetic-fusion-003
```

Это таблицы по доступному snapshot; они не доказывают полноту исходного запуска.
`temporal_fusion.py` оценивает только synthetic VAL и сохраняет причины
неполных/несовместимых групп в `coverage.csv`.

## Проверки переноса

CPU-тесты покрывают авторизацию сохранённых карт по хешам, запрет записи в
исходные каталоги, неизвестную GT, COCO/геометрию, matched support, LOFO,
сочетания, классовые срезы, GeoSCD retry/resume и CLI из другого каталога.

```bash
python -m unittest discover -s tests -p 'test_real_*.py' -v
python -m unittest discover -s tests -p 'test_geoscd_real_retry.py' -v
python -m unittest discover -s tests -p 'test_temporal_fusion.py' -v
```
