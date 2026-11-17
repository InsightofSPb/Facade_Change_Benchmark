# Проверка сокращённого history-scenarios — 2026-11-09

Проверенная исходная база: `50ff7adcc3d25ec26fd1b43440bcb1b702b5c489`.
Работа выполнена в той же ветке `codex/2026-11-08-history-three-scenarios`.
CPU: Python 3.12, Torch 2.15.0+cpu, NumPy 2.2.6; использовано готовое
окружение `/workspace/work/venv`, без загрузки реальных данных или DINO weights.

## Выполненные проверки

Полный завершённый проход: **1298 тестов, OK, 7 skipped**, 396.867 s.
После исправления записи frozen JSON отдельно прошли **11 runner/training тестов**.
Последний повтор полного suite остановлен по прямой просьбе владельца;
его результат не объявляется успешным завершённым прогоном.
Новые проверки после этой просьбы не запускались.

Команда обязательного набора:

```bash
/workspace/work/venv/bin/python -s -B -m unittest discover -s tests -v
```

Первый проход обнаружил eager TEST-access в legacy stack-original при повторной
адаптации metadata в `summarize_histograms`. Исправление оставляет вычисление
сводных метрик операцией над готовыми histogram/category данными. Второй сбой
первого прохода — ожидаемая защита MSDZip resume от изменения Python source tree
во время параллельной разработки; финальная проверка выполнялась без правок кода.

Профильные тесты покрывают точные 18/48/8, все пять fusion-голов и backward,
пустую поддержку, заморозку входов, реальный короткий fit всех типов голов,
маски/равный вклад зданий, CNN halo и canonical overlap, причинность прогнозов,
F1-only patience, top-3 и уникальные вторичные победители, семь границ прерывания
checkpoint transaction, точный optimizer/scheduler/scaler/RNG resume,
выборочное чтение/dedup, scoped cache preparation без TEST, preflight gate,
ресурсные ограничения, независимые сравнения с контролем и stack, residual
в главном VAL-отборе, primary-checkpoint NLL и read-only отчёт.

Датированные тесты включаются существующим `load_tests` в `tests/test_data_analysis.py`;
обычный `unittest discover` не пропускает их из-за дефисов в имени.

## Сквозной synthetic запуск

Используется тот же конфиг и матрица, с явно обозначенным режимом `smoke`:
пространственный размер `smoke`, 2 primary эпохи, 1 fusion эпоха,
effective batch=2, без AMP. Это уменьшенная техническая fixture, не оценка
реальных `large` моделей или научного выигрыша.

Получено 18 completed primary, 48 completed score-fusion, 8 completed
feature-fusion. 87 оцениваемых методов = 74 обученных + 8 standalone residual
+ 5 fixed references. Для 12 событий сохранены 1044 event-map descriptors,
по 348 на TRAIN/VAL/TEST. Основной VAL-отбор содержит 9 вариантов; главная таблица
дополняется stack, двумя copy-last и отдельной строкой semantic state A.
Подробная таблица содержит 87 строк; per-event/class/group CSV — 13 572 строки.

Первый CLI-run проверен независимым аудитом: решения главной таблицы и пороги
зафиксированы в ledger до первого TEST artifact; controls соответствуют голове,
прогнозные NLL используют основной checkpoint и quantizer=0.1.
Проверены hashes 2833 файлов schema-2 descriptors/arrays и всех checkpoints.
Все 1789 численных массивов конечны: float32 либо исходный точный dtype,
без float16. Физический объём NPY этой маленькой fixture — 885 176 bytes;
это измерение fixture, не прогноз объёма реального набора.

## Совместимость и отсутствие перезаписи

Legacy schema-1 report-only выполнен на отдельной полной копии прежнего
`2026-11-08-scenarios-smoke-008`. Исходные 7869 файлов проверены SHA-256 до и после:
изменённых файлов нет. Старый run, provenance и source caches не мигрировались.

## Preflight и ресурсы

Вызов нового `preflight` с реальными путями владельца завершился
`NOT_MEASURED`: `/home/sasha/...` и CUDA отсутствуют в Cloud. Сохранены JSON/text
resource report и причина; primary/fusion обучения не запускались.

Отдельный `preflight --synthetic --device cpu` измеряет 54 записи
model/history-role (с явным reuse совпадающих квантилей) для 18 primary IDs
и пять групп fusion с отображением на 48+8. Он сохраняет `SYNTHETIC_MEASURED`,
`ready_for_full=false`. Guard-тесты запрещают научные TEST reads и проверяют
восстановление RNG. Отдельно проверены просроченный fingerprint, нехватка
памяти/места и сохранение частичного resource report при OOM.

Измерены только Cloud CPU-операции на synthetic данных: IO, шаги обучения,
VAL aggregation/threshold/metrics, запись, frozen-head inference, report-only,
gallery и RSS. Несжатый постоянный/временный объём и полные бюджеты эпох —
оценки по явным формулам, не замеры полного benchmark. CUDA peaks, реальные
распределения длин/форм и длительность эксперимента владельца остаются
`NOT_MEASURED`. Сжимаемость случайных прогнозов не переносится на обученные.

CLI-флаги проверены `python -m data_analysis history-scenarios --help`.
Полные команды для владельца и пути артефактов находятся в
[протоколе](2026-11-09-history-three-scenarios-reduced.md#команды-владельца).
Результаты synthetic не подтверждают пользу истории; реальный TEST остаётся
exploratory, а building bootstrap не оценивает вариативность seeds.

## Итоговая фиксация

CLI smoke `2026-11-08-reduced-smoke-verified-004` завершён на итоговой
численной реализации: все 18+48+8 обучений, 1044 карты. Проверенный ранее
resume/report-only сохранил hashes 4434 карт, весов и связанных научных файлов;
report-only и произвольная TEST-визуализация выполнялись с запрещёнными
конструкторами моделей, inference и загрузкой checkpoints.

Расширенный аудит выявил перезапись `training_records`/TEST coverage; исправление
проверено regression test с hashes и mtime, без model/GT calls. Затем устранено
различие сериализации `resolved_matrix.json`: frozen контракты проверяются
и не переписываются. Последний отдельный CLI resume после этой малой правки
не запускался: владелец попросил прекратить тестирование и отправить код.

Измеренный пиковый RSS synthetic CPU preflight: 465113088 bytes.
Его оценки постоянного/временного несжатого объёма: 33188088/4390476 bytes
для маленькой fixture; переносить эти числа на реальные фасады нельзя.

Конфиг и новые документы датированы 2026-11-09 по Europe/Moscow.
Имена существующих модулей и совместимых артефактов сохраняют 2026-11-08.
После проверок изменены только дата/ссылки нового конфига и документов.
