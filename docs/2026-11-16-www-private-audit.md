# WWW: приватный архивный аудит и blind factual review, 2026-11-16

Подготовка review на CPU. Модели, промпт, бюджеты, retrieval и строгая валидация
ответов не меняются. Сравнение C−A и общий prompt revision из `c2f00fc` сохранены.
Ни одна метка, alias или eligibility не утверждается этим экспортом.

Для нового RGB QA review действует отдельная, явно включаемая
[политика `rgb-qa-2026-11-16-v2`](2026-11-16-www-rgb-qa-policy.md).
Описанный ниже запрет пересечения segmentation splits сохраняется для legacy
policy; новая политика записывает его как факт и отдельно проверяет QA exposure.
Identity/asset/coordinate validation и pending approvals остаются прежними.

## Что проверено и что пока недоступно

После `c169a28` исправлена совместимость с сообщённой реальной схемой:
плоские source metadata в asset register и plural identifiers в proposals.
Обе ссылки Library (исходный пакет и V4.1 supplement) по-прежнему возвращают HTTP
403. В Cloud проверены **синтетические** fixtures с этой схемой, собственными RGB
и честно вычисленными SHA; полный реальный пакет здесь ещё не проверен.

Независимая диагностическая нормализация реального пакета, по сообщению владельца,
проверила 41 изображение и 39 панелей без ошибок. Ожидаемая структура: 13 pending
строк, восемь строк ссылаются на семь существующих вопросов, пять shortlist rows
остаются unresolved. Fixture воспроизводит эти количества; импортёр их не навязывает.

Владелец также подтвердил реальный `www audit`: **48 строк = 44 completed,
3 error, 1 abstained**. Отсутствующие изображения при независимой проверке были
связаны с недоступными WSL-путями. Эта правка не меняет references, factual audit
или inference; повторный инференс для исправления путей не нужен.

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
  и либо непустой `provenance`, либо реальные плоские поля: `label`, `evidence_id`,
  `capture_date`, `precision`, `source_url`, `note`, `source_workspace_path`,
  `delivery_location`. Плоские source fields копируются во внутренний
  `provenance.source_fields` с меткой `schema:flat_asset_register`. Все исходные
  поля, включая дополнительные, остаются в записи, а register сохраняется
  byte-for-byte. Пустой provenance не подставляется; требуется непустой source
  locator/evidence ID. Переписывать исходный ZIP не нужно;
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

### V4.1 — дополнение к оригинальной галерее

`2026-11-16-v4.1-correction-private-handoff.zip` (около 62 КБ) содержит mappings
и comparison cards, но не изображения галереи. Передавать его через
`--private-audit-supplement` вместе с
`--private-audit .../2026-11-16-www-v4-private-region-review.zip`.
Флаг поддержан в `archive-audit-review`, `review-page` и `review-package`.
Попытка использовать supplement вместо оригинального пакета выдаёт явную ошибку.

Все файлы дополнения, независимо от имени/суффикса, доступны на отдельной странице
`supplement/gallery.html`, перечислены в `audit.private.json:supplement_files`
с SHA и сохранены byte-for-byte под инертными `.txt` именами. HTML показывается
как экранированный текст; бинарные файлы явно отмечены как доступные без preview.
На `index.html` есть ссылка на этот раздел и скачивание каждого исходного файла.
Это **явное предоставление для review**, а не автоматическое применение исправлений:
`status:exposed_pending_not_applied`, `supplemental_coordinates_applied:false`.
Даже совпадающее имя register/proposals в supplement не перезаписывает оригинал.
Mappings не объявляются verified; projected boxes не создаются, все отображённые
рамки остаются на точном native asset. Для будущего переноса требуется отдельная
проверка mapping, обоих SHA и обеих решёток. Никаких approvals этот шаг не даёт.

## Identity, кандидаты и scope

`Voskova_10/Voskova_12` и `baskov_32/baskov_34` остаются pending в существующем
identity mapping. Подтверждённая человеком физическая группа с пересечением
оригинальных splits исключается **и из development, и из evaluation**. Для Voskova
это известное последствие VAL/TEST overlap, а не повод ослабить guard. Summary и
pending gallery объясняют правило; исходные IDs, splits, exposure и inventory
сохраняются. При передаче inventory используется тот же `--identity-map` по SHA/пути.

Связь с candidate устанавливается только по точным supplied question ID либо
view+region ID и однозначной записи manifest. Принимаются прежние скаляры
`building_id/view_id/region_id/question_id` и реальные массивы
`inventory_ids/local_view_ids/local_region_ids/question_ids` соответственно.
Внутри массива IDs — альтернативы; между полями ограничения должны согласоваться.
Скаляр и массив одного поля также должны согласоваться. Несколько оставшихся
кандидатов дают `ambiguous:true`, без выбора первого; конфликт IDs не обходится.
`building_group` остаётся proposed display label, а не canonical building ID.
Исходный `category` сохраняется и не превращается в утверждённую метку.

Точная ссылка помечается `context_only:true`: существующий вопрос служит контекстом
и не утверждает другой предложенный вопрос, его wording или scope. Исходный вопрос
не заменяется текстом comparison card. Адрес, похожая архитектура, даты или alias
не заменяют проверку соответствия. При отсутствии IDs показываются
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

Положить **настоящие** скачанные ZIP в `/home/sasha/Downloads/`:
`2026-11-16-www-v4-private-region-review.zip` и
`2026-11-16-v4.1-correction-private-handoff.zip`, либо заменить каталог фактическим.
Это реальные имена пакетов, сообщённые владельцем; наличие их в Downloads не предполагается.
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
  --private-audit /home/sasha/Downloads/2026-11-16-www-v4-private-region-review.zip \
  --private-audit-supplement /home/sasha/Downloads/2026-11-16-v4.1-correction-private-handoff.zip \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-review-original-16-audit-004 \
  2>&1 | tee /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-review-original-16-audit-004.console.log
```

Открыть `.../2026-11-16-www-review-original-16-audit-004/review.html`; отдельная
ссылка ведёт в `pending-audit/index.html`. Все shortlist rows останутся доступными,
даже если исходный candidate manifest не позволяет разрешить их exact IDs.
Этот раздел использует V3.1 явно; несовпадение заявленного private corpus SHA
помечается `different_revision_pending_recheck`, без подмены версии.

## Foreground WSL: настоящий 48-row pilot, без инференса

Реальный экспорт уже выполнен владельцем: 44 completed / 3 error / 1 abstained.
Для этой правки повторять его не требуется. Только если нужен новый offline export
на WSL с доступными исходными изображениями, использовать отдельный каталог:

```bash
cd /home/sasha/Facade_Change_Benchmark &&
git pull --ff-only &&
set -o pipefail &&
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -u -m facade_change.agentic www audit \
  --run /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-models-16-001 \
  --candidates /home/sasha/Facade_Change_Benchmark/runs/2026-11-15-facade-qa-candidates-001/candidates.json \
  --handoff /home/sasha/Facade_Change_Benchmark/runs/2026-11-15-www-handoff-001/handoff.private.json \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-factual-audit-16-002 \
  2>&1 | tee /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-factual-audit-16-002.console.log
```

Открыть `.../2026-11-16-www-factual-audit-16-002/factual-review.html` после нового
экспорта. Существующий `...-001` сохраняется; ссылки и references не переписываются
ради отсутствующих в Cloud WSL-путей. Cloud эту реальную команду не выполнял.

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
  --private-audit /home/sasha/Downloads/2026-11-16-www-v4-private-region-review.zip \
  --private-audit-supplement /home/sasha/Downloads/2026-11-16-v4.1-correction-private-handoff.zip \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-review-all-v41-002 \
  2>&1 | tee /home/sasha/Facade_Change_Benchmark/runs/2026-11-16-www-review-all-v41-002.console.log
```

Не считать 100-case cohort готовой до фактического inventory/selection и ручного
review. Для shortlisted событий всё ещё нужны доступные датированные исходники,
точные candidate IDs, provenance, human identity/correspondence и scope decisions.

## Выполненные CPU-проверки

Для правки совместимости после `c169a28`, в существующей Cloud CPU-среде,
без установки пакетов:

```bash
/workspace/work/facade-venv/bin/python -s -B -m unittest \
  tests.test_2026_11_16_www_audit.PrivateArchiveAuditTests \
  tests.test_2026_11_16_www_review \
  tests.test_2026_11_15_www_curation -v
```

**28 проверок прошли.** В том числе: 41 flat asset / 39 panels / 0 panel errors;
13 pending / 8 linked rows / 7 unique questions / 5 unresolved / 0 approvals;
неизменные source fields, register bytes и context questions; отказ на неверном
SHA, размерах, формате и пустой provenance; ambiguity/conflicting IDs; plural
inventory IDs с прежним cross-split guard; все supplemental files, включая
нестандартные имена, HTML и binary, видимы/сохранены без override и projection;
сохранение human text, отдельного original-16 workflow и paired C−A.

Сквозной `www review-page` с обоими синтетическими ZIP и handoff также прошёл:
те же counts, два видимых supplemental файла, неизменный human target на основной
странице и в linked context. stdout — корректный JSON, прогресс — stderr.
WSL-блоки проверены `bash -n`. Реальная схема supplemental файлов недоступна:
они намеренно предоставляются как исходный материал, а не объявляются проверенными mappings.

Исторические 60 CPU + 2 JS checks исходного экспортера относятся к `c169a28`.
Его interrupted 48-row fixture имел другие synthetic statuses; это не результат
настоящего пилота владельца (44/3/1).

Полный repository suite и GPU не запускались. Проверка настоящих private packages,
candidate shortfalls и публичного V4.1 по-прежнему требует доступных WSL-файлов. Результатов
о качестве моделей или готовности реальной 100-case когорты эти проверки не дают.
