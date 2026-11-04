# Полный экспорт кропов и геометрически пригодные истории

Оцениваем TRAIN, VAL и TEST раздельно. TRAIN нужен для диагностики, галерей
и выбора порогов; он может содержать снимки, использованные при обучении
Semantic DINO. Его результат не оценивает перенос на новые фасады. Известные
пересечения и непроверенное происхождение checkpoint сохраняют свой статус:
название VAL/TEST само по себе не доказывает независимость от всех старых весов.

Это два разных набора наблюдений:

- Полный парный экспорт содержит все сохранённые окна и пары дат. Размер,
  шаг и геометрическая поддержка берутся из готового crop dataset.
- Полные пригодные истории содержат окна в общей системе координат всех
  принятых дат вида. Они требуют first-anchor geometry и общей поддержки.
  Их число не равно числу независимых пар из экспорта.

## Парные кропы

Используйте существующую рабочую среду. Команды выполняются из
`/home/sasha/Facade_Change_Benchmark`. Подготовленные владельцем маски:

```bash
python -s -u -m facade_change workflow prepare targets \
  --crops data/crops \
  --splits train val test \
  --out runs/2026-11-04-full-targets-001
```

Если этот запуск уже завершён, повторять его не нужно. Незавершённую подготовку
продолжают той же командой с `--resume`. RGB и alignment не пересоздаются.

CPU-проверка RGB/SSIM уже завершена: по 20 пар из каждого split. Следующий
шаг — пилот всех девяти методов на одних и тех же 5 парах из каждого split,
затем один общий полный запуск. Отдельные полные прогоны групп методов не нужны.

Сначала восстановите настройки готовых моделей из сохранённого эксперимента.
Этот шаг читает provenance, проверяет наличие и SHA-256 файлов моделей,
включая MSDZip ABS/MOD256 и Semantic DINO. Обучение, скачивание весов и inference
не запускаются. Старый prepared используется только как источник настроек
моделей; новый набор пар по-прежнему определяется `data/crops`.

```bash
python -s -u - <<'PYCONFIG'
import json
from pathlib import Path

from scripts.real_eval.night_common import (
    DEFAULT_METHODS,
    load_prepared,
    read_model_contracts,
)

_, manifest = load_prepared("runs/2026-10-17-night-prepared-001")
options = {}
for method in DEFAULT_METHODS:
    print(f"Checking {method}", flush=True)
    contract = read_model_contracts(manifest, [method], device="cuda:0")[method]
    options[method] = contract["options"]

output = Path("configs/2026-11-04-methods-full-eval.local.json")
with output.open("x", encoding="utf-8") as stream:
    json.dump({"methods": options}, stream, indent=2, ensure_ascii=False)
    stream.write("\n")
print(f"Saved: {output}")
PYCONFIG
```

Новый конфиг не перезаписывает `configs/methods.local.json`; повторное создание
того же файла также запрещено. Если он уже успешно создан, используйте его.
Путь старого prepared относится к компьютеру владельца. MSDZip использует
сохранённый `worker_python`, когда он указан, как и прежний эксперимент.

Пилот всех методов:

```bash
time python -s -u -m facade_change workflow benchmark crops \
  --crops data/crops \
  --targets-run runs/2026-11-04-full-targets-001 \
  --splits train val test \
  --methods rgb_diff ssim lpips dinov2 semantic_dino anychange geoscd msdzip_abs msdzip_mod256 \
  --methods-config configs/2026-11-04-methods-full-eval.local.json \
  --max-pairs-per-split 5 \
  --device cuda:0 \
  --out runs/2026-11-04-all-methods-pilot-001
```

Ошибка `msdzip_abs requires checkpoint_path in --methods-config` происходит
до создания run record и вычисления карт. После именно этой ошибки можно
повторить команду с тем же `--out` без `--resume`. Если запуск уже успел создать
`run.json`, применяются обычные правила продолжения ниже.

Пилот выбирает пары детерминированно по ID, без просмотра масок или оценок.
Он проверяет вычисления и даёт первые замеры времени; его метрики не заменяют
полный результат. Если в выбранном TRAIN нет известных положительных или
отрицательных пикселей, калибровка остановится с пояснением. Тогда увеличьте
лимит с новым `--out`; автоматического отбора по наличию изменений нет.
Проверьте завершение всех методов, `summary.json`, `metrics.json`, `cases.csv`
и `gallery.html`, оцените время общего прогона. Затем запустите все методы
на полном экспорте:

```bash
time python -s -u -m facade_change workflow benchmark crops \
  --crops data/crops \
  --targets-run runs/2026-11-04-full-targets-001 \
  --splits train val test \
  --methods rgb_diff ssim lpips dinov2 semantic_dino anychange geoscd msdzip_abs msdzip_mod256 \
  --methods-config configs/2026-11-04-methods-full-eval.local.json \
  --max-pairs-per-split 0 \
  --device cuda:0 \
  --out runs/2026-11-04-full-crops-all-methods-001
```

Для переноса готовых весов MSDZip на полный экспорт проверяются сохранённые
TRAIN/VAL идентификаторы зданий: обучение на текущем VAL/TEST или выбор
checkpoint на текущем TEST блокирует запуск. Совпадение TRAIN допускается
и остаётся диагностикой. Это проверка стабильных ID зданий, а не доказательство
независимости всего внешнего предобучения. Старый режим оценки аугментаций
сохраняет точную проверку fingerprint своего датасета.
ArIB/BCM в новом режиме полного экспорта пока не поддерживаются.

### Протокол и результаты

- `--crops` выбирает весь экспорт указанных split. Без лимита остаются в том
  числе неизменившиеся и полностью неизвестные по аннотациям окна.
- `--dataset-run` сохраняет прежний benchmark выбранных аугментированных
  TRAIN/VAL-примеров. Его лимит `0` не означает полный исходный экспорт.
- Один порог на метод выбирается только на TRAIN, затем фиксируется для VAL
  и TEST. Native-mask методы сохраняют собственное бинарное предсказание.
- На вход scorer поступают RGB и геометрическая поддержка. Аннотации нужны
  только оценщику. Unknown=255 не становится отрицательным классом.
- F1/IoU изменений, precision/recall и FPR сохраняются вместе с TP/FP/FN/TN,
  числами известных/игнорируемых пикселей и поклассовыми результатами.
  Полностью неизвестное окно имеет нулевое оценочное покрытие, а не идеальный F1.
- Сначала усредняются пригодные наблюдения здания, затем здания получают равный
  вес. TRAIN/VAL/TEST не объединяются в основную итоговую метрику. Перекрывающиеся
  кропы не считаются независимыми фасадами и не превращаются в IoU всей стены.
- `selection.json` содержит фактически включённые ID и численности;
  `threshold_selection.json` — пороги и TRAIN-кривые; `cases.csv` и `classes.csv`
  — результаты наблюдений; `metrics.json` — агрегаты; `maps/` — карты методов.

Прерванный benchmark продолжается той же командой с `--resume`. Нельзя менять
методы, split, лимит или веса под прежним `--out`. Завершённые запуски неизменяемы.
Проверка кодом и CPU-примером не подтверждает качество на реальных фасадах.

## Истории PREVIOUS / EARLIEST / HISTORY

В `prepare real` добавлен отдельный режим полного геометрического покрытия.
Он не ранжирует окна по изменениям, не требует положительных или известных
GT-пикселей и не ограничивает число фасадов/окон. Старый GT-enriched режим
остаётся доступен для воспроизведения прежних результатов.

Вход — завершённый актуальный `real-audit`, включающий TRAIN, VAL и TEST и
нужную сессию выравнивания. Создание аудита требует исходного frozen experiment,
его snapshot и текущего semantic dataset; эти пути берутся из существующих
run records, не из парного crop index. Команда `prepare real-audit` описана в
[основном workflow](2026-10-24-main-workflow.md#истории-и-robustness).

После подготовки такого аудита в `data/history_audit`:

```bash
python -s -u -m facade_change workflow prepare real \
  --audit data/history_audit \
  --selection-mode all-geometric \
  --out runs/2026-11-04-full-histories-001
```

Проверьте `manifest.json`, численности по split и `gallery.html`. Галерея может
показывать ограниченное число историй; этот предел не сокращает манифест.
Для сохранения прежнего пространственного протокола используются окна 512×512
с шагом 256; это отдельный опыт от парных кропов 224×224.

Далее прежние готовые модели читают расширенный манифест:

```bash
python -s -u -m facade_change workflow benchmark pilot \
  --prepared runs/2026-11-04-full-histories-001 \
  --device cuda:0 \
  --out runs/2026-11-04-full-histories-pilot-001

python -s -u -m facade_change workflow benchmark real \
  --prepared runs/2026-11-04-full-histories-001 \
  --modes previous earliest history \
  --reuse-run runs/2026-11-04-full-histories-pilot-001 \
  --device cuda:0 \
  --out runs/2026-11-04-full-histories-benchmark-001
```

Этот runner сохраняет исходные веса, пороги и правило причинной памяти своего
frozen experiment. Его калибровка не заменяется TRAIN-калибровкой парного runner.
Семантические маршруты и directed-delta анализ запускаются на том же новом
манифесте; карты прежних отдельных ROI нельзя приписывать полному покрытию.
Оценка охватывает пригодную геометрию и известную аннотацию, а не гарантированно
всю поверхность каждого фасада.
