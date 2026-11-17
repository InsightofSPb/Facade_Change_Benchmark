# Аудит и консолидация dev_late — 2026-11-17

Задача: собрать активные разработки и сократить повтор логики и инструкций,
сохранив научные протоколы. Результат строится от RAG `bf4f4c47`; старые ветки,
данные, веса и результаты не удаляются. Все проверки ниже используют синтетику
и CPU; это инженерное подтверждение, не новый научный результат.

## Исходный граф и решения по веткам

Refs проверены на GitHub 2026-11-17. `dev_late` отсутствовала. В Cloud не было
checkout владельца или его данных. Созданы отдельные source, baseline и
`dev_late` worktrees; чужих изменений не было.

| Ветка | SHA | Решение |
| --- | --- | --- |
| main | `9233021ba06769dd83c8bd3be7fb3efc29c911ec` | Предок RAG |
| codex/2026-10-18-facade-atlas | `4f1c71451788a587716ea2dbc1b107686469aae9` | Предок RAG |
| codex/2026-10-18-semantic-routes | `5315a64e369f716fa62f6af7268a1344e2097ebe` | Предок RAG |
| codex/2026-10-19-dataset-statistics | `b5bdb7b8680974a563bd871d01d813177e863ab1` | Исключена: у владельца более новая недоступная локальная версия |
| codex/2026-10-19-semantic-update-eval | `a65f75a916ab0ad25e1ec3669520812e221a0884` | Предок RAG |
| codex/2026-10-24-main-workflow | `c31b82951e991f5ba38ce58c49b541644996324b` | Memory follow-ups и transition bank перенесены через history-ветку; файлы в двух источниках совпадают |
| codex/2026-10-25-spatial-change | `ba58767b68dd76a39aa01716938839e30984062a` | Предок RAG |
| codex/2026-11-08-change-agentic-rag | `bf4f4c47e321259a8023812776c21252c006fd27` | Основа с последними WWW fixes |
| codex/2026-11-08-history-three-scenarios | `bf7e8b4ff8a0b20f1ff03db50e2fa1c56c056781` | 35 новых файлов: история, storage, preflight, модели, отчёты, конфигурации, регрессии |
| codex/2026-11-10-adaptive-visual-inspection | `30a3ed81eff5f35bee46491036bc8c49387417ce` | 13 новых файлов; общие CLI/QA-файлы сверены отдельно |
| codex/fix-night-pilot-provenance | `7dc74797da0a90bb8da76bcae07bcbdcc5649aa4` | Предок RAG |
| codex/night-real-val-lofo | `b93ec1cbdfe0c8f0e2cb38f10262a9d918b99980` | Предок RAG |

Merge-base RAG/history: `b2d3acebfc27eabba3955f2366ee0c8e2c9e1689`;
RAG/inspection: `9aa159c0489e3a4ee15a089b1e214ade8d980fc3`.
Ветки целиком не сливались. `agentic/models.py` уже совпадал в RAG и inspection,
поэтому resource verification не заменялся второй реализацией.

Общие QA-изменения сохранены совместно: WWW archival labels/date intervals,
inspection review-chain hashes и принадлежность building, проверки physical
identity. Экспорт WWW сохраняет порядок candidate cohort; inspection явно
задаёт `preserve_selection_order=True`. Повторные/неизвестные subset IDs
отклоняются, сохраняется отдельная provenance-запись выбора.

## Владельцы и устранённые повторы

Полная навигационная таблица находится в
[указателе компонентов](2026-10-24-main-workflow.md); здесь только решения.
Поиск учитывал imports, importlib, subprocess, workflow dispatch, configs,
тесты и документированные команды. Дата файла не служила критерием удаления.

| Общая операция | Было | Стало / потребители |
| --- | --- | --- |
| Проверка физической идентичности | Копии в WWW и inspection, расходящиеся по pending-policy | Один validator `agentic/2026-11-16_www_identity.py`; inspection — тонкий subclass с явным `pending_candidates=True`; WWW inventory/retrieval/evaluation сохраняют свою policy |
| Строгий JSON transport | Повторные hooks для duplicate keys, NaN/Infinity и overflow | `io.strict_json_decoder`; WWW runner и inspection reply используют его, сохраняя разные fence/answer schemas |
| Binary precision/recall/F1/IoU/FPR | Повтор формул и edge-case политики в real_gt_metrics и real_gt_slices | `real_gt_metrics._metrics`; class-slice wrapper добавляет только marked_fraction. ROI, ignore, thresholds и aggregation не менялись |
| Датированные тесты анализа | Часть существующих fusion/memory tests не попадала в обычный unittest discovery | `test_data_analysis.load_tests` подключает существующие component suites и перенесённые history tests; inspection включён в `test_workflow.load_tests` |
| Агентский вход | Pair-маршруты workflow и отдельный agentic CLI, без общего маршрута к WWW/inspection | Рекомендуемый `workflow agentic …` вызывает тот же backend; прежние входы совместимы |

RGB scoring/RemotePool/registry, geometry, annotation projection, галереи и
RunStore уже имеют общие реализации. Они сохранены. Ledger не добавлялся:
`SharedMapStore` истории хранит массивы, `RegionStore` — наблюдения и evidence,
а lifecycle обоих сценариев опирается на существующий RunStore.

Не объединены разные научные evaluators: semantic state, synthetic H0/H1,
clean-TRAIN crop benchmark, VAL history benchmark, LOFO, supplied-region QA и
archive answer evaluation. Политики пустого GT/нулевых знаменателей у них
различаются. Не переносились старые COCO counts, ontology IDs или split из памяти.

Тонкие atlas/bundle/report launcher-ы уже вызывали пакет; повторного переноса
не требовали. Их локальные потребители владельца недоступны, поэтому совместимые
имена сохранены. Остальные scripts/real_eval отвечают за отдельные действующие
протоколы; массовый перенос изменил бы frozen source identities без доказанной
пользы этой задаче. Старые research/codec-направления сохраняются как явно
ограниченные протоколы, а не как обязательная цель новой master-map.

## Оркестрация и расширение

Существующий workflow расширен маршрутом `agentic` и просмотром dispatch:
`workflow --dry-run STAGE TASK …`. Dry-run не импортирует backend, не пишет run,
не запускает подготовку, модель или запрос; печатает выбранный модуль и все
аргументы, включая явные входы/выходы. Проверяет ограничения стадии; проверка
совместимости данных/model hashes остаётся у backend до inference.
Dry-run намеренно обозначен как `dispatch_only`, а не как успешный preflight.

`workflow run --config` не добавлен: это было условное предложение задачи.
Сценарии уже содержат проверенные фазовые команды, но не общий контракт
передачи результата. Тонкий новый маршрут не вводит auto-latest, автоматическое
обучение или обещание resume, которого нет у backend.

Методы подключаются через существующий registry; датасеты — через контракт
соответствующей задачи и provider/input adapter. Fake-adapter и независимые
CPU-fixture проверки сохранены. CityPulse/STVchrono и остальные кандидаты карты
не объявлены реализованными адаптерами или пройденной внешней валидацией.

## Master-map и документация

Единственный канонический файл:
`docs/2026-11-17_dissertation_research_master_map.md`.
Преобразование DOCX сохранило текст и порядок всех 714 абзацев, 15 таблиц и
29 гиперссылок; разделы, оговорки, даты, численные значения и банк идей не сокращались.
SHA-256 исходного DOCX:
`8370c39e4b2a68c3300cb6ae611eb7f4d9f9aaab47aeddf1d732f7ace7585d13`.
SHA-256 Markdown:
`a5be7be1cacc73abe35e217d4d5f1e266dcca5c11b6bb904f6eeb22b7f968e1b`.
Старая августовская карта не определяет текущие решения.

README содержит пять сценариев; AGENTS — правила изменения; существующий
workflow-документ стал одним указателем владельцев, configs, tests и runbooks.
Предыдущая подробная история переноса доступна по неизменяемой ссылке на `bf4f4c4`.
В исходных runbooks отмечено историческое назначение прежних branch checkout-команд.

| Проверяемая сложность | До | После |
| --- | --- | --- |
| Источники активных возможностей | RAG, main-workflow, history, inspection | Один checkout dev_late; происхождение четырёх веток сохранено |
| Реализации трёх объединённых операций | По две в исходных ветках/модулях | По одному владельцу, policy остаётся явной |
| README / AGENTS / workflow-index | 237 / 74 / 434 строки, включая историю и повторные инструкции | 70 / 63 / 158 строк с направленными ссылками |
| Основной путь к нужному компоненту | README с экспериментами и множеством отдельных ссылок | README → одна строка указателя → код/тест; пять маршрутов приведены в указателе |
| Публичные совместимые входы | Несколько имён для ряда операций | Сохранены, где локальные потребители неизвестны; один рекомендуемый агентский маршрут |

Число строк здесь — проверяемое описание навигации, не критерий качества.
Число файлов увеличилось из-за переноса уникальных активных возможностей.
Экономия токенов не измерялась и не заявляется.

## Проверки

Исходный системный Python не имел OpenCV/pycocotools и имел Pillow вне
поддерживаемого диапазона. Первый диагностический suite: 1235 tests,
24 failures, 56 errors, 196 skips. После `scripts/setup_cloud.sh` в отдельном
venv исходная RAG: 1420 tests, OK, 126 skips. Для CPU-тестов перенесённой истории
установлен `torch==2.5.1+cpu`; эта же CPU-версия закреплена в
`requirements-cpu.txt`, который уже используют Cloud setup и CI. Это зависимость
полного тестового окружения; базовые зависимости пакета не менялись. Веса
моделей не скачивались. В той же среде
повтор исходной RAG: 1420 tests, OK, 10 skips.

Команда полного сравнения: `python -s -B -m unittest discover -s tests -v`.
Среда: Python 3.12; NumPy 2.2.6, Pillow 11.3.0, OpenCV headless 4.12.0.88,
pycocotools 2.0.11, SciPy 1.13.1, tqdm 4.67.1, PyTorch 2.5.1+cpu.

Перенос истории: 178 tests, OK. Первичная интеграция workflow/inspection:
69 tests, OK. Финальный полный dev_late suite: **1641 tests, OK, 10 skips**,
553.827 s. Из них 1631 выполнены, failures/errors нет.
В первом общем прогоне найден неверный путь общего `io.py` в WWW source-hash
contract. Исправлено относительное имя `../io.py`; проверка не ослаблена.
Регрессия frozen smoke → continuation после исправления проходит.
Также восстановлен standalone-import метрик для старых прямых CLI: 12 focused
checks проходят. Первый общий dev_late прогон: 1641 tests, 2 failures,
2 errors, 10 skips; две ошибки относятся к standalone CLI, одна — к пути
JSON dependency, ещё одна — ожидаемый отказ source guard, поскольку исправление
попало во время теста. Финальный suite выполнен на неизменяемом коде.

Дополнительно выполнены реальные CLI smoke: SIFT/SSIM/LZMA на трёх датированных
RGB-наблюдениях (два перехода, ноль ошибок) и inspection demo (пять вопросов,
четыре политики, ответы/отказы, exposure, HTML и offline evaluation). Модельные
адаптеры inspection подставные; эти числа не измеряют научное качество.

Точный паритет двух checkout на одних входах, без численного допуска:

- 72 строки RGB-DIFF/SSIM: thresholds, TP/FP/FN/TN, score/prediction hashes;
- шесть синтетических историй: все последовательности/contexts и 56 PNG;
- пять архивных вопросов × три режима: 15 ответов, 15 exposure-записей,
  одинаковые offline metrics; scripted pool явно моделирует ответ, отказ и ошибку;
- все 299 исходных файлов остались побайтно неизменными; HTML/metrics созданы.

SHA-256 нормализованного parity-артефакта до и после:
`3a1630830e703ae67340f6ee59362c1c5f1cb5238a5cf511ba27cb90343f6faa`.
Исключены только несопоставимые provenance/пути отдельных выходов и время;
сравниваемые карты, решения, IDs, хронология и метрики не нормализовались.

## Ограничения

Реальные фасадные данные, локальная dataset_statistics, CUDA, настоящие
Qwen/SAM3/Jina/DINO и веса внешних моделей в Cloud не проверялись.
Source hashes изменились: прежние frozen runs воспроизводятся на своих commits,
а не через ослабление проверок новой ветки. Большие эксперименты и API-вызовы
не запускались. Нового вывода о научном качестве или преимуществе агента нет.

Необязательные skips среды: Transformers runtime (3); ArIB original compiled
source (1); BCM+VTM author runtime (1); BCM author checkout (1); zstandard в
трёх разных codec/roundtrip-проверках (3); sklearn AP parity (1).
Эти пропуски не являются подтверждением работы соответствующих моделей.
