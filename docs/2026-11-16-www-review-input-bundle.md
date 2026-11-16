# Пакет исходных данных для подготовки заполненных карточек

`scripts/2026-11-16-package-www-review.py` собирает текущую выбранную когорту
из существующего `review-package`, ничего не отбирая заново и не утверждая метки.
В ZIP входят исходные candidates/selection, метаданные review, онтология и
порядок битовых каналов, packed-маски, RGB-кропы, геометрическая поддержка,
исходные полные кадры и геометрия всех наблюдений выбранных регионов.
Остальная история этих регионов — только приватный контекст проверки;
набор изображений в вопросах для A/B/C остаётся прежним.

Сохраняются исходные байты и SHA-256. Пути не переписываются в manifests:
отдельный `archive_paths` связывает пути WSL с файлами ZIP. Файлы с одинаковыми
байтами упаковываются один раз. Проверяется и содержимое во время записи ZIP.
Неизвестные или отсутствующие аннотации не превращаются в фон/отсутствие дефекта.

V4.1 media повторно не включаются, если они уже доступны получателю. Для проверки
привязки передаётся сам public corpus JSON. Полные исходные кадры выбранных
историй включаются: поэтому ZIP может быть заметно больше набора одних кропов.
Веса моделей, inference outputs и весь каталог `runs` не копируются.

Старые `runs/2026-11-15-www-handoff-001/{handoff,selection}.private.json`
при наличии сохраняются отдельно, без переноса на новые question IDs.
Экспортированные decisions из каталога подготовки и `checked-*/reviewed.json`
также копируются без автоматического применения. Для дополнительного файла
используйте повторяемый `--decisions /полный/путь.json`.
Несохранённое состояние открытой HTML-страницы команда прочитать не может.

```bash
cd /home/sasha/Facade_Change_Benchmark &&
git switch codex/2026-11-08-change-agentic-rag &&
git pull --ff-only &&
set -o pipefail &&
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -u \
  /home/sasha/Facade_Change_Benchmark/scripts/2026-11-16-package-www-review.py \
  --candidates /home/sasha/Facade_Change_Benchmark/runs/2026-11-15-www-candidates-all-001/candidates.json \
  --review-package /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-home-v41-001/review \
  --archive-corpus /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-home-v41-001/archive/corpus.json \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-review-inputs-001.zip \
  2>&1 | tee /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-review-inputs-001.console.log
```

Если preparation folder называется иначе, замените только соответствующие пути.
При успехе выводятся `status=completed`, фактическое число вопросов, регионов,
файлов, размер ZIP, `models_executed=false`, `automatic_approvals=0`.
При несовпадении SHA/выборки, отсутствии обязательного файла или существующем
`--out` команда завершается с объяснением. Незавершённый ZIP нельзя использовать.

Три CPU-проверки: перенос настоящих fixture RGB/NPZ/full frames и исходного
manifest без изменения байтов; отказ при чужом selection SHA; отказ при
изменённом RGB. Повторная запись готового ZIP также запрещена.
