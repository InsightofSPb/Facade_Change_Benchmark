# Первые проверки оплаченных H0/H1 и контекста опоры

Два ограниченных эксперимента по мастер-карте исследования (§8, §28).
Они проверяют механизм и полезность контекста на существующих процедурных
TRAIN/VAL; это не готовая модель истории фасада и не оценка реальных повреждений.
Прежние результаты benchmark и карты TEST не изменяются.

## Первый запуск

Из корня репозитория, в существующей среде `scd_bench`. Новые веса не скачиваются,
BCM/VTM не запускаются. Путь к датасету, обученному оригинальному MSDZip mod256
и локальным DINOv2 исходникам/весам читается из проверенного завершённого run.

```bash
(
  cd /home/sasha/Facade_Change_Benchmark &&
  git pull --ff-only &&
  conda activate scd_bench &&
  FC_RESEARCH_TAG="$(date -u +%Y%m%dT%H%M%S%NZ)" &&
  python -u -s -m facade_change.research both \
    --source-run "$PWD/runs/20261010T131625089776Z-neural-codecs-comparison" \
    --max-buildings-per-split 1 \
    --max-bases-per-building 1 \
    --scenarios-per-base 1 \
    --epochs 1 \
    --max-train-bytes 2048 \
    --max-val-bytes 512 \
    --device cuda:0 \
    --trust-checkpoint \
    --out "$PWD/runs/${FC_RESEARCH_TAG}-research-smoke"
)
```

`--trust-checkpoint` относится только к ранее использованным локальным MSDZip/
DINO checkpoint и нужен для legacy Torch. Restricted loading пробуется первым.
Не передавайте этим флагом доверие произвольному скачанному файлу.

Это smoke: один фасад TRAIN и один VAL, одинаковый сценарий для unchanged,
crack и paint_patch на каждом выбранном кропе. По шести случаям нельзя делать
вывод о качестве. Цель — проверить загрузку настоящих локальных моделей,
выходные артефакты и время на домашней машине. В Cloud оригинальный MSDZip и
обучение heads проверены на CPU; provider DINO заменён фикстурой,
авторские DINO-веса и CUDA здесь не запускались.

После успешного smoke — фиксированный малый эксперимент, снова в новом каталоге:

```bash
FC_RESEARCH_TAG="$(date -u +%Y%m%dT%H%M%S%NZ)"
python -u -s -m facade_change.research both \
  --source-run "$PWD/runs/20261010T131625089776Z-neural-codecs-comparison" \
  --max-buildings-per-split 3 \
  --max-bases-per-building 1 \
  --scenarios-per-base 2 \
  --epochs 3 \
  --max-train-bytes 32768 \
  --max-val-bytes 8192 \
  --device cuda:0 \
  --trust-checkpoint \
  --out "$PWD/runs/${FC_RESEARCH_TAG}-research-feasibility"
```

Выбор сохраняется до чтения RGB/GT: SHA256 ranking, до трёх фасадов в каждой
части, один кроп на фасад, до двух сценариев на кроп. Все три состояния
получают те же сценарии. Это одинаковая выборка для обоих экспериментов, но не
повтор старого quick VAL/TEST. Наличие TRAIN обязательно; TEST не заменяет TRAIN.
Нулевая квота означает весь доступный набор, поэтому не используйте её в smoke.

`paid` и `context` запускают только соответствующий эксперимент. Если paid
завершился, а context завершился ошибкой, исправьте установленную причину и
запустите `context` в новом каталоге: повторять paid не требуется. Старые
результаты не перезаписываются, автоматического resume нет.
Для другого датасета можно передать `--dataset-run` и явные
`--msdzip-checkpoint`, `--dino-root`, `--dino-checkpoint`.

## A. Окупается ли оплаченная RGB-аппроксимация обновления

Один готовый кодер `zlib` (level9) используется для обоих mod256 остатков.
H0 выбирает identity либо RGB-affine prediction; все оценённые по текущему RGB
параметры передаются в потоке. H1 имеет ту же nuisance family и дополнительно
платит за битовую маску C, флаг операции и три RGB-байта V. V задаёт fill цветом
либо modular shift внутри C. Остаток восстанавливает каждый поддержанный
RGB-байт точно. H0/H1 независимо выбирают минимальную полную стоимость
из фиксированных режимов. GT и synthetic nuisance parameters в кодер не входят.

Первый рабочий кандидат C — RGB-DIFF >0.08; этот фиксированный proposal
полностью оплачивается. `proposal_coverage.csv` показывает его собственный Recall
и FPR: итоговый Recall не может превысить покрытие C. Отдельно кодируется
`oracle_visible_GT_paid` (файлы с суффиксом `oracle_paid`): C = labels==1.
Это GT-диагностика оплаченного представления, не работающий детектор и не
математическая верхняя граница. Oracle не участвует в калибровке и сравнении
качества. Следующий этап — другие GT-free кандидаты из замороженных моделей;
они не подменяются GT в этом первом прототипе.

Для каждого перекрывающегося тайла вычисляется реальное ΔL=L0−L1.
Карта хранит max ΔL/support_pixel среди тайлов, предложивших этот пиксель в C.
Пиксели вне C не могут стать изменением; τ>=0 не допускает отрицательной
экономии H1. Декодер получает только A, shared reference support и поток.
Проверяются точные round trips обоих потоков и oracle на каждом тайле.
Заголовки, sha256, индексы режимов, длины секций, nuisance, C, V, residual
входят в полную длину. Их сумма совпадает с `len(stream)*8`.

Lossless scope — RGB на заранее общей геометрической поддержке опоры.
Целевые RGB вне неё не кодируются; декодер оставляет там A. Это не whole-image
compression rate: тайлы перекрываются и имеют самостоятельные потоки.
Поддержка кодирования не сокращается с помощью GT visibility/occlusion.

Этот прототип намеренно ограничен: RGB fill/shift не представляет онтологические
переходы или историю S→C,V. Identity/affine не объясняют произвольные тени,
blur и окклюзии. Отрицательный результат относится к этой family/update format,
а не автоматически ко всей диссертационной гипотезе.

## B. Помогает ли прошлый контекст и добавляет ли что-то спектральная ветка

Три режима используют один frozen оригинальный MSDZip checkpoint, его causal
byte windows, lanes и 256-значный словарь без изменений upstream:

| Режим | Вероятности RGB-остатка |
|---|---|
| baseline | исходные logits MSDZip |
| spatial | исходные logits + обучаемый head по frozen DINO-признакам опоры |
| spectral | исходные logits + head по amplitude/phase этих же признаков |

Spatial/spectral имеют одинаковые две token-attention ветки, число параметров,
initial weights, Adam, seed, число эпох и sampled bytes. Финальная logit-коррекция
инициализируется нулём: исходные вероятности точно сохраняются до обучения.
Обучаются только heads на H0 TRAIN. Каждый checkpoint выбирается по минимуму
H0 VAL NLL; после этого модель замораживается для карт всех трёх состояний.

DINOv2 ViT-S/14 загружается через существующий local-only loader. В features
попадает только опорный RGB, с прежней нормализацией и reflect padding;
неподдержанные пиксели детерминированно обнуляются. Никакие target features,
labels, nuisance parameters, Fourier residual или будущие target bytes
не передаются в head. Native RGB mod256 residual остаётся исходным сигналом.

Spectral head использует полный rFFT spectrum и обе ветви amplitude/phase,
без удаления частот. Это собственная небольшая адаптация идеи SET
(https://arxiv.org/abs/2407.18568), не авторская реализация SET внутри слоёв VFM
и без его attention optimization. Пространственный контроль отделяет эффект
спектрального представления от capacity. Сравнение с baseline также меняет
наличие обучаемого адаптера и само по себе не доказывает эффект именно контекста.

Здесь выводятся **model bits / NLL**, без neural entropy stream и без заявления
о проверенном декодировании. Быстрый эксперимент не требует VTM или
нейросетевого сжатия/разжатия; actual roundtrip сейчас относится к A/zlib.

## Метрики, артефакты и решение о продолжении

Первичная метрика — facade-macro visible-H1 pixel Recall при VAL H0 pixel
FPR <=1% и <=5%. τ выбирается только по VAL: максимум Recall, затем меньший
FPR и более высокий τ. Строгое сравнение float32 `score > threshold`.
GT255 игнорируется. Средние — случаи внутри фасада, затем равные веса фасадов.
TRAIN-качество описательное; VAL используется и для checkpoint selection,
и для калибровки. Это не независимый окончательный тест.

Дополнительная диагностика `inspection_budget.csv` ранжирует неперекрывающиеся
32px тайлы по max score и показывает покрытие видимых H1 при номинальных
5/10/25% тайлов. GT не участвует в ранжировании. Реальная доля выбранной
пиксельной поддержки записывается отдельно из-за округления и крайних тайлов.
Это покрытие областей для осмотра, а не FPR маски изменений.

В `.../paid/` и `.../context/` сохраняются:

- `run.json`, `experiment_contract.json`, `events.jsonl`, `selected_cases.json`,
  `input_artifacts.json`: immutable fresh run, configuration/input/source hashes,
  append-only events и конечный статус;
- `summary.json` / `summary.txt`, `threshold_selection.json`, `case_metrics.csv`,
  `facades.csv`, `classes.csv`, `inspection_budget.csv`;
- paid: `tile_costs.csv`, `proposal_coverage.csv`, `oracle_diagnostics.csv`,
  `streams/*.bin`, `scores/*/*.npy`. Bundle содержит последовательные
  самостоятельные потоки; offsets/sizes и stream SHA записаны в tile_costs;
- context: `sampling.json`, `training_history.csv`, `model_provenance.json`,
  `reference_preparation.json`, `spatial.pt`, `spectral.pt`, `raw_bpb/*/*.npy`
  и `scores/*/*.npy`.

`full_diagnostic_seconds` paid включает oracle и все проверки декодера.
Для operational runtime потребуется отдельный замер без oracle.
Новая попытка — новый каталог; частичные файлы при ошибке сохраняются.

После одного фиксированного малого прогона:

1. Посмотреть, какие потери Recall идут от proposal C, а какие — от ΔL gate;
   отдельно crack/paint и каждый фасад. Проверить окупаемость oracle paid C,V.
2. Сравнить spatial/spectral при том же FPR и inspection budget. Падение H0 NLL
   полезно лишь вместе с сохранённым Recall. После трёх фиксированных эпох
   не начинать перебор гиперпараметров на уже просмотренном TEST.
3. При сохранённом Recall и устойчивом эффекте перейти к другим fixed candidates
   и настоящему причинному контексту ≥3 дат после инвентаризации таких серий.
   Неуверенность сегментации остаётся будущим флагом маршрутизации, не veto.

## Проверка кода

```bash
python -m unittest discover -s tests -p 'test_paid_codec.py' -v
python -m unittest discover -s tests -p 'test_reference_context.py' -v
python -m unittest discover -s tests -p 'test_research_experiments.py' -v
```

CPU fixtures проверяют настоящие original MSDZip logits и optimizer steps,
zero-head parity, capacity match, causal native offsets, FFT/padding,
декодирование/битовые расходы, независимость обычных потоков от GT,
VAL-only калибровку по фасадам и отсутствие доступа к TEST RGB.
Кодовые тесты не доказывают научный эффект.
