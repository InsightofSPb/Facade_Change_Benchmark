# Facade Change Benchmark

Исследование изменений исторических фасадов: исходные RGB и архивы →
совмещение → карты состояния и изменения → история → ответ с показанными
свидетельствами. Реальные данные, веса и результаты хранятся локально.

Научный план — [master-map от 2026-11-17](docs/2026-11-17_dissertation_research_master_map.md):
D — данные, X — внешний перенос, V — визуальные методы, R — агентский RAG.
Карта содержит планы и гипотезы; их наличие не означает реализованный метод.
[Указатель компонентов и проверок](docs/2026-10-24-main-workflow.md) ведёт от
операции к владельцу кода, конфигурации и одной подробной инструкции.

## Пять рабочих сценариев

Команды выполняются из корня checkout в существующей среде проекта.
Для новой Cloud-среды зависимости и CPU-проверка: `bash scripts/setup_cloud.sh`.
Исследовательскую среду владельца не переустанавливайте.

1. **Проверить RGB-путь на CPU без данных и весов.** Три процедурных наблюдения,
   настоящий SIFT/SSIM/LZMA, история и HTML:

   ```bash
   python -m facade_change workflow agentic smoke --kind cpu --out runs/2026-11-17-cpu-001
   ```

   Для метрик RGB-DIFF/SSIM на подготовленном контролируемом наборе:
   `python -m facade_change workflow benchmark synthetic --dataset-run /absolute/path/to/h0h1-run --methods rgb_diff ssim --out runs/2026-11-17-baseline-001`.
   [Подготовка и ожидаемые артефакты](docs/2026-10-24-main-workflow.md#rgb-baseline).

2. **Подготовить данные или обучить сегментацию.**
   `python -m facade_change workflow prepare --help` и
   `python -m facade_change workflow train segmentation --help`.
   [Данные, split и сегментация](docs/2026-10-24-main-workflow.md#данные-и-сегментация).
   Обучение — отдельный явный шаг.

3. **Сравнить пары и истории.**
   `python -m facade_change workflow benchmark real --help`;
   для сохранённых признаков и экспериментов истории —
   `python -m data_analysis history-scenarios --help`.
   [Истории и оценка](docs/2026-10-24-main-workflow.md#истории-и-оценка).

4. **Проверить агентские инструменты и показ evidence.**
   `python -m facade_change workflow agentic --help` объединяет существующие
   pair, QA, WWW и inspection-команды. Полный офлайн-пример с подставной
   моделью: `python -m facade_change workflow agentic www demo --out runs/2026-11-17-www-demo-001`.
   [Инструменты, поиск и ответы](docs/2026-10-24-main-workflow.md#агентские-сценарии).

5. **Построить отчёт по сохранённому запуску.**
   `python -m facade_change workflow gallery --help`.
   Для истории: `python -m facade_change workflow gallery real --run /absolute/path/to/history-run --out runs/2026-11-17-report-001`.
   Отчёты и анализ используют явный исходный run.

`facade workflow ...` после установки эквивалентно `python -m facade_change workflow ...`.
`workflow --dry-run STAGE TASK ...` печатает маршрут и аргументы без запуска
backend; проверка данных и моделей выполняется его `check`/`preflight` или
валидацией перед исполнением. Это просмотр команды, не подтверждение готовности.
Каждому новому запуску нужен новый `--out`; resume доступен лишь у поддерживающего
его backend. Полного автоматического `workflow run` нет.

## Работа с кодом

Начните с [AGENTS.md](AGENTS.md), затем выберите строку
[указателя компонентов](docs/2026-10-24-main-workflow.md).
Полный CPU suite: `python -s -B -m unittest discover -s tests -v`.
CPU fixtures проверяют исполнение и контракты; качество моделей на реальных
зданиях требует отдельного эксперимента.

[Аудит и проверка dev_late от 2026-11-17](docs/2026-11-17-dev-late-audit.md)
фиксирует исходные SHA, переносы и ограничения. Исторические документы сохраняют
протоколы своих запусков; текущие маршруты выбираются через указатель компонентов.
