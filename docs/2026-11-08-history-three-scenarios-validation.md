# Проверка реализации эксперимента истории

Дата: 2026-11-08. База: `c31b82951e991f5ba38ce58c49b541644996324b`.
Ветка: `codex/2026-11-08-history-three-scenarios`.

## Что проверено

Cloud: Python 3.12, NumPy 2.2.6, CPU PyTorch 2.15.0+cpu. Зависимости установлены
из `requirements-cpu.txt` в отдельное виртуальное окружение; существующее
окружение владельца не изменялось. Веса DINO и реальные изображения не скачивались.

Полная обязательная команда:

```bash
python -s -B -m unittest discover -s tests -v
```

Результат: **1227 tests, OK, skipped=7**, 398.368 s. Датированные тесты включены
в обычный discovery через `tests/test_data_analysis.py`: сам unittest пропускает
имена файлов с дефисами. Последующие локальные изменения runner/report проверены
отдельными фокусными тестами и пересборкой сохранённого сквозного запуска.

Причины семи пропусков: отсутствуют original ArIB author checkout/compiled coder;
BCM/VTM encoder/decoder/config и original BCM checkout; необязательные zstandard
и sklearn. Новые архитектуры, обучение, сценарии A/B/C и report-only проверены
с установленным CPU Torch, а не пропущены.

Фокусная команда:

```bash
python -m unittest \
  tests.test_2026-11-08_history_scenarios_models \
  tests.test_2026-11-08_history_scenarios_data \
  tests.test_2026-11-08_history_scenarios_abc \
  tests.test_2026-11-08_history_scenarios_training \
  tests.test_2026-11-08_history_scenarios_report \
  tests.test_2026-11-08_history_scenarios_runner -v
```

Проверены формы, padding, support, конечные градиенты, реальное время,
пересчёт signed/absolute delta, удаление текущих/будущих данных из прогнозного
preprocessing, отсутствие GT/detector/semantic access в forecast, TRAIN-only
нормализация, ontology/split/cache integrity, точная вероятность quantization bin,
G, состояния A, pair-only B, progression proxy C, исходный AnyChange OR при overlap,
пороговый evaluator, NA/insufficient/error, изоляция ошибок моделей и checkpoint resume.

Основные размеры ConvGRU/Transformer дополнительно прошли forward/backward
на `[1,3,384,16,16]`, включая direct и predictive головы; H0 использует тот же
predictive head и отдельно проверен обучением с H0-loss. Восемь сочетаний дали
конечные выходы и градиенты. Один измеренный CPU-шаг без optimizer/IO занимал
0.120–0.556 s при одном потоке; это не прогноз времени локального обучения.

## Сквозной синтетический запуск

Последний проверенный полный synthetic run: `2026-11-08-scenarios-smoke-008`,
статус `completed`. Он прошёл основной CLI `history-scenarios smoke --device cpu`:
47 конфигураций smoke-матрицы, все со статусом `completed`, 2676 float-map units,
19 строк главной таблицы, отдельные prediction/semantic/classwise/building
метрики, сравнительные таблицы и фиксированная/error gallery. В главной матрице
реального запуска остаются **60 основных вариантов + 11 A/B/C** с budget 60 эпох;
проверочные размеры/эпохи smoke её не заменяют.

CPU-preflight и фактический `--help` также выполнены. `compileall` и
`git diff --check` прошли. При ручном просмотре synthetic gallery проверены
история с годами, temporal GT/unknown, stack/direct, residual/U/G и TP/FP/FN.

Затем выполнены `resume` и `report-only` с monkeypatch-запретом создания моделей
и forward. У **7550 проверяемых файлов** (NPZ, checkpoint, unit markers и
зафиксированные контракты) SHA-256 сохранились; главная таблица совпала побайтно.
Report-only дополнительно выполнялся с запрещёнными dataset constructors.
Это подтверждает отсутствие повторного обучения/inference и чтения исходного
датасета в данном сохранённом завершённом запуске.

После ограничения памяти отчётного этапа этот же набор пересобран повторно:
главная CSV-таблица совпала побайтно, хеши всех 2676 scientific maps сохранились.
Измерение report-only: 20.24 s, peak RSS 222556 KiB (около 217.3 MiB) на данном
synthetic fixture. Гистограммы хранятся во временном дисковом буфере и читаются
по одному методу/split; тесты проверяют очистку собственных temporary files
при успехе и ошибке. Это измерение не обещает тот же RAM/runtime на реальных данных.

Последняя фокусная проверка включает 67 тестов новых модулей. Изменения после
полного suite затрагивают ресурсные записи runner и ограничение памяти reporter;
для них выполнены отдельные регрессии, повторный export/report-only integration
и проверка неизменности сохранённых результатов.

Отдельный тест прерывает обучение после завершённой эпохи: после resume model,
optimizer, scheduler и RNG дают тот же результат, что непрерывное обучение.
Проверены отказ при изменении нормализации, хеши checkpoint/fusion и восстановление
CUDA warn-only policy после внутреннего строгого CPU smoke.

## Ограничения и принятые решения

- Реальные пути `/home/sasha/...`, изображения, исходные runs и GPU владельца
  в Cloud отсутствуют. Реальный cache/split/checkpoint preflight и полное обучение
  выполняются локально. Научная польза истории этой проверкой **не установлена**.
- Совместимость существующего spatial DINO cache проверена по контракту кода;
  реальные hashes и доступность устанавливает локальный preflight. Отбор исходной
  canonical cohort ретроспективный, хотя прогноз по фиксированному прошлому причинен.
- Новые методы и общий stack сравниваются на одной feature-compatible поддержке.
  Это более узкая область, чем прежний raw canonical support; `stack-original`
  сохраняется отдельно для проверки исходных counts. Unknown не получает score
  unchanged для включения в метрики.
- A checkpoint выбирается по downstream VAL-критерию; semantic-state качество
  этого checkpoint выводится отдельно. Отдельный оптимум semantic checkpoint
  в этой реализации не заявляется.
- C использует заранее записанный annotation-derived proxy и проверку достаточности
  данных; отсутствие размеченной динамики не заменяется положительной синтетикой.
- Архитектуры реализованы на PyTorch по принципу факторизованного attention;
  чужие training frameworks/видео-checkpoints не импортированы. Предобучен только DINO.
- Исторические реальные числа не подставлены из задания. При наличии старых
  файлов runner реконструирует доступные показатели из per-building counts;
  отсутствие этих артефактов фиксирует как unavailable.
- Checkpoint payload полной матрицы при всех 60 эпохах оценивается примерно
  в 156 GiB без карт и A/B/C. Preflight сохраняет локальные RAM/VRAM/disk и
  измерения максимальной TRAIN-длины; срок завершения без этих измерений не обещается.

Команды владельцу и точные основные выходы приведены в
[протоколе запуска](2026-11-08-history-three-scenarios.md).
