# WWW: приватный архивный аудит и blind factual review, 2026-11-16

Подготовка review на CPU. Модели, промпт, бюджеты, retrieval и строгая валидация
ответов не меняются. Сравнение C−A и общий prompt revision из `c2f00fc` сохранены.
Ни одна метка, alias или eligibility не утверждается этим экспортом.

## Что проверено и что пока недоступно

Ссылка Library на новый приватный пакет вернула HTTP 403; сам пакет и настоящий
48-row pilot в Cloud недоступны. Импорт проверен на **синтетической** структуре:
13 предложений (7 region audits и 6 shortlisted comparisons), отдельная фотография
1110×640 и другая публичная страница 1013×1432 под связанным evidence ID.
Это не проверка реальных изображений, строк пакета или ответов Qwen.

Все 16 объектов решений в ранее приложенном
`2026-11-15-www-human-review-handoff.private.json`, включая уточнение case 03,
сравнены с сохранённым CPU handoff: совпадают. Проверка их привязки к оригинальным
candidate bytes на WSL остаётся обязательной; команды ниже проверяют SHA.

## Приватный пакет и точная принадлежность координат

`www archive-audit-review` читает локальный каталог или ZIP, а `review-page` и
`review-package` вызывают тот же экспорт через `--private-audit`. Все предложения
доступны в отдельном pending-разделе, даже если eligible batch пуст. Точно связанные
кандидаты открываются в существующем редакторе вопросов/ответов; это доступ для
review, а не включение в когорту.

Поддерживаемый вход (проверен на fixture, реальные JSON ещё нужно сверить):

- `2026-11-16-review-proposals.private.json`: список строк либо объект с `rows`
  или `proposals`; непустой уникальный `review_id`, список `panels`;
- `2026-11-16-asset-register.private.json`: mapping asset-key → запись либо
  `{"assets": mapping}`. Запись содержит `sha256`, целочисленные `width`, `height`
  и `provenance`;
- `2026-11-16-side-by-side-gallery.private.html`: один inert
  `<script id="asset-data" type="application/json">` с mapping asset-key →
  PNG/JPEG data URL либо `{"data_url": ...}`;
- вспомогательные alias/derivative/locator/audit `.private.json/csv/txt/md`
  сохраняются и показываются как текст. Скрипты не исполняются; исходный HTML
  сохраняется с окончанием `.html.txt`, а review строится своим шаблоном.

Проверяются реальные SHA, размеры и формат каждого asset. При несовпадении
импорт останавливается. `panel.asset` определяет исходную решётку; необязательная
`box_xyxy` — четыре конечных native pixel coordinates `[left, top, right, bottom]`
с положительными шириной/высотой и границами внутри этого asset. Поле
`coordinate_system`, если указано, равно `asset_native`. Необязательные
`asset_sha256`/`image_sha256` также проверяются. Ошибочная панель сохраняется с
конкретной причиной и исходными координатами; её рамка не исправляется и не рисуется.

Предоставленный владельцем пример: `survey-p46-photo2`, **1110×640**,
SHA `71f5cc65595b59ebe4aba53f4abef462d441b72acc2dfbb321dc5991bec931a3`, и
публичный raster **1013×1432** для `ev-ee02a969ee16fe87` — разные изображения.
Реальные байты этого примера в Cloud не проверены. Синтетическая картинка имеет
собственный вычисленный SHA; приведённый реальный SHA ей не присваивается.

Боксы рисуются на зарегистрированном источнике без EXIF-поворота. Evidence ID
сам по себе не переносит координаты на другое изображение. Этот импортёр не
реализует перенос даже при наличии заявленного `transform`: он требует отдельно
проверенного преобразования и оставляет исходную панель с диагностикой. Derivative
provenance сохраняется. Correspondence proposals всегда отделены от event GT;
рамки приватного extracted photo не подставляются в редактор публичной PDF-page.

## Identity, кандидаты и scope

`Voskova_10/Voskova_12` и `baskov_32/baskov_34` остаются pending в существующем
identity mapping. Подтверждённая человеком физическая группа с пересечением
оригинальных splits исключается **и из development, и из evaluation**. Для Voskova
это известное последствие VAL/TEST overlap, а не повод ослабить guard. Summary и
pending gallery объясняют правило; исходные IDs, splits, exposure и inventory
сохраняются. При передаче inventory используется тот же `--identity-map` по SHA/пути.

Связь с candidate устанавливается только по точным supplied question ID либо
view+region ID и однозначной записи manifest. Адрес, похожая архитектура, даты или
alias не заменяют проверку соответствия. При отсутствии IDs показываются
доступные кандидаты и `missing_inputs`, без выдуманной связи. Приоритеты ручного
разрешения остаются такими:

| Сравнение | Предполагаемый тип, ещё не утверждённый | Недостающая проверка |
| --- | --- | --- |
| Baskov 32: local 2013 / archive 2021-04-20 | plaster loss | точные view/region/question, локальный кроп, correspondence |
| Bolshoy 83: 2008-04-20 / 2014-06-04 | upper parapet alteration | точный регион и датированные archive assets |
| Kamennoostrovsky 67: 1934 / 2004-09-07 | corner-roof alteration | точный регион, источник дат и correspondence |

Архитектурная переделка не становится автоматически повреждением или ремонтом.
Earliest editor требует явного человеческого выбора `supplied_observations` либо
`frozen_collection` и уточнения текста при новом/изменённом scope. Исходные вопросы
не переписываются: это новая reviewed revision. Одна и та же версия вопроса и
references затем замораживается для A/B/C; старые labels остаются совместимыми.

Цели 40 earliest / 30 visible-change / 20 stability / 10 insufficient-evidence,
physical-building balancing, overlap/exposure exclusions и отчёт о недоборе
остаются прежними. Отбор не читает model outcomes. В fixture с нулевой eligibility
недобор равен 100, но все 13 pending-предложений доступны. Это **не** подсчёт
допустимых вопросов владельца.

## Offline factual review сохранённого запуска

`www audit --run ...` не запускает модели, не требует approved answers/plan и не
создаёт retrospective plan для старого unscored run. Он привязывается к исходному
execution contract и frozen questions. Экспортирует все question × condition,
включая error, missing и abstained; отсутствующие строки не исключаются.

Свидетельства восстанавливаются из сохранённых Qwen request blocks, включая
показанные, но не процитированные источники. Ровно переданный `source_excerpt`
не дополняется скрытым полным текстом. Изображение должно совпасть с SHA/путём
exposure log. Retrieval после последнего Qwen-вызова не считается показанным.
Если requests/байты отсутствуют, это явная диагностическая неполнота, а не
придуманная экспозиция. Сохранённый request доказывает передачу worker, но сам по
себе не доказывает успешную обработку при ошибке worker.

`factual-review.html` и `blind-review.private.csv/json` показывают вопрос, ответ,
последний raw response и submitted evidence. Salted blind IDs перемешивают порядок;
соответствие question/condition и технические costs/пути находятся отдельно в
`blind-key.private.json`. По самим фото/цитатам условие иногда можно догадаться:
blind export не обещает скрыть различия в доступных свидетельствах.

Человек отдельно вводит `target_event_match`, `claim_support` и
`citation_support`, reviewer и notes. Пустое поле остаётся **unreviewed**. Экспорт
суждений привязан к `source_review_id`. Совпадение класса «change», наличие citation
ID и допустимый bbox не проставляют factual correctness. Опциональный
`--candidates ... --handoff ...` показывает исходный человеческий target verbatim
только после проверки исходных candidate bytes и публичного вопроса. Это не новая
метка и не автоматическая оценка. Без target reviewer должен явно отметить
неразрешённую идентичность события, если её нельзя установить.

Формальный `www score` по-прежнему требует заранее замороженный plan; его
C−A/B−A/C−B, building bootstrap и знаменатели не менялись. Он переиспользует тот
же новый factual exporter. `www audit` сам не вычисляет accuracy и не заменяет
полный аудит корректности старых ответов/контракта исполнения.

## Foreground WSL: исходные 16 и private audit

Положить **настоящий** скачанный private ZIP в
`/home/sasha/Downloads/2026-11-16-www-private-archive-audit.zip` либо заменить этот
единственный путь фактическим. Наличие файла по этому имени не предполагается.
Не запускать содержимое пакета. Все пути ниже абсолютные; shell-global variables
не используются. Если output уже существует, выбрать новый номер, не удалять старый.

```bash
cd /home/sasha/Facade_Change_Benchmark &&
git pull --ff-only &&
set -o pipefail &&
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -u -m facade_change.agentic www review-page \
  --candidates /home/sasha/Facade_Change_Benchmark/runs/2026-11-15-facade-qa-candidates-001/candidates.json \
  --selection /home/sasha/Facade_Change_Benchmark/runs/2026-11-15-www-handoff-001/selection.private.json \
  --handoff /home/sasha/Facade_Change_Benchmark/runs/2026-11-15-www-handoff-001/handoff.private.json \
  --archive-corpus /home/sasha/Facade_Change_Benchmark/runs/2026-11-15-www-archive-v31-001/corpus.json \
  --private-audit /home/sasha/Downloads/2026-11-16-www-private-archive-audit.zip \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-review-original-16-audit-003 \
  2>&1 | tee /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-review-original-16-audit-003.console.log
```

Открыть `.../2026-11-16-www-review-original-16-audit-003/review.html`; отдельная
ссылка ведёт в `pending-audit/index.html`. Все shortlist rows останутся доступными,
даже если исходный candidate manifest не позволяет разрешить их exact IDs.
Этот раздел использует V3.1 явно; несовпадение заявленного private corpus SHA
помечается `different_revision_pending_recheck`, без подмены версии.

## Foreground WSL: настоящий 48-row pilot, без инференса

```bash
cd /home/sasha/Facade_Change_Benchmark &&
git pull --ff-only &&
set -o pipefail &&
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -u -m facade_change.agentic www audit \
  --run /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-models-16-001 \
  --candidates /home/sasha/Facade_Change_Benchmark/runs/2026-11-15-facade-qa-candidates-001/candidates.json \
  --handoff /home/sasha/Facade_Change_Benchmark/runs/2026-11-15-www-handoff-001/handoff.private.json \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-factual-audit-16-001 \
  2>&1 | tee /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-factual-audit-16-001.console.log
```

Открыть `.../2026-11-16-www-factual-audit-16-001/factual-review.html`. Число 48 —
ожидание 16×3; фактические статусы, неполнота и сохранённые outputs проверяются
при выполнении команды. Сейчас этот реальный экспорт **не выполнен**.

## V4.1 и отдельный all-candidates review

V4 и старые execution contracts остаются неизменными. После появления public
metadata-only V4.1 импортировать настоящий package в **новый** каталог через
`archive-import`. Его `corpus.json` получит фактический SHA; private audit записывает
его отдельно от заявленного SHA в proposals. Несовпадение означает pending recheck,
не разрешение перепривязать старые решения. В новый inference contract попадёт
SHA нового корпуса/индекса, а не придуманная константа для V4.1.

All-candidates подготовка остаётся отдельной от original-16, как описано в
[предыдущем обновлении](2026-11-16-www-ca-v4-preparation.md). Переиспользовать manifest
можно лишь при совпадающих histories/annotations/freeze/availability. Не передавать
старый handoff новому manifest; передавать оригинальный public16 через
`--exposed-questions`. Для уже подготовленных файлов следующая команда — только
новый CPU review, без approvals и без индексации. Пути `...v41-001` ниже обозначают
будущие outputs, которые должны сначала реально появиться; их наличие не заявлено.

```bash
cd /home/sasha/Facade_Change_Benchmark &&
set -o pipefail &&
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -u -m facade_change.agentic www review-package \
  --candidates /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-candidates-all-v41-001/candidates.json \
  --histories /home/sasha/Facade_Change_Benchmark/data/histories \
  --archive-corpus /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-archive-v41-001/corpus.json \
  --exposed-questions /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-public-16-001/questions.json \
  --private-audit /home/sasha/Downloads/2026-11-16-www-private-archive-audit.zip \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-review-all-v41-001 \
  2>&1 | tee /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-review-all-v41-001.console.log
```

Не считать 100-case cohort готовой до фактического inventory/selection и ручного
review. Для shortlisted событий всё ещё нужны доступные датированные исходники,
точные candidate IDs, provenance, human identity/correspondence и scope decisions.

## Выполненные CPU-проверки

В существующей Cloud CPU-среде, без установки пакетов:

```bash
/workspace/work/facade-venv/bin/python -s -B -m unittest \
  tests.test_2026_11_16_www_audit \
  tests.test_2026_11_16_www_review \
  tests.test_2026_11_15_www_curation \
  tests.test_2026_11_15_www_archive_review \
  tests.test_2026_11_10_facade_questions -v
node tests/test_2026_11_16_www_review_ui.js
node tests/test_2026_11_16_www_factual_ui.js
```

60 Python checks прошли; проверены exact-asset/native bounds, отказ от переноса,
SHA/dimension mismatch, private/public boundary, pending/zero batch/cross-split,
сохранение human text и старой цепочки review, paired C−A и новые factual exports.
Два Node DOM checks проверяют ручной earliest scope и экспорт только введённых
суждений. Это не browser/GPU/model-quality validation.

Синтетический interrupted 16×3 fixture экспортировал 48 строк: 39 completed,
3 error, 3 abstained, 3 missing; все human judgments пустые. Отдельно проверены
shown-but-uncited evidence и retrieved-but-never-submitted evidence. CLI
`archive-audit-review` прошёл на 13-row fixture, а `audit` — на ранее сохранённом
9-row scripted run из `c2f00fc`, без запуска нового inference и без retrospective
plan. Итоговые stdout — JSON, прогресс — stderr. WSL-блоки проверены `bash -n`.

Полный repository suite не запускался. Настоящий private package, реальный 48-row
pilot, candidate shortfalls и данные V4.1 ожидают доступных WSL-файлов. Результатов
о качестве моделей или готовности реальной 100-case когорты эти проверки не дают.
