# WWW: C−A, архивная проверка и подготовка V4 — 2026-11-16

Продолжение `28a8e6d`. Независимая работа с кодом завершена; реальные данные и
результаты владельца в Cloud недоступны. Подготовка ниже не утверждает, что
100 вопросов уже отобраны, review пройден или аудит 48 строк завершён.

## Что известно о реальном 16-question run

По подтверждённым сведениям владельца C выполнил **ноль retrieval calls**.
A и C получили одинаковые system instructions, вопросы и фотографии;
различался заключительный control message: `final`/0 rounds против
`retrieve_or_final`/2 rounds. Qwen3.5-4B работал с `do_sample=false`,
`enable_thinking=false`. Следовательно, входы A/C не полностью идентичны,
а различие ответов при нулевом retrieval нельзя приписывать архивным источникам.

Дополнительное совпадение C по классу change описывало кондиционер, тогда как
человеческий reference — участок ремонта. Это **classification agreement,
не factual correctness**. Structured correctness и macro-F1 по классам остаются
узкими метриками. Фактическая поддержка, совпадение описанного события и корректность
цитат по смыслу требуют blind human review; exposed IDs не являются доказательством.
Эти сведения получены от владельца, не реконструированы здесь из raw traces.

## Изменения кода и контракта

В offline scoring добавлен C−A рядом с B−A и C−B для всех существующих
development/evaluation и all/archival strata. Используются те же frozen
questions/references, physical-building clusters, 2000 resamples, seed=20261116,
95% percentile intervals и прежний учёт errors/missing. Aliases не считаются
независимыми зданиями. Для change/stability применён тот же binary reference
subset; insufficient references не превращаются в stability.

Старая инструкция C описывала формат retrieve и остаток бюджета, но не задавала
критерий необходимости архивной проверки. Минимальное уточнение находится в
**общем SYSTEM всех A/B/C**:

- Показанных фото достаточно для наблюдаемого различия на заданных датах или
  самого раннего свидетельства **среди показанных наблюдений** — допустим local-only final.
- Для вывода за пределами этих наблюдений или исторического факта, не видимого
  на фото, нужны архивные свидетельства. Если их нет, поиск разрешён и остался
  бюджет — запросить их целевым поиском.
- Если поиск недоступен/не помог, ограничить вывод показанными наблюдениями либо
  признать недостаточность. Не объявлять локальную раннюю фотографию началом
  события или самым ранним свидетельством во всём архиве.

Это инструкция модели, а не новый принудительный dispatch: C по-прежнему может
завершить без поиска. Формат control message, инструменты, фильтры, строгая
валидация дат/boxes/citations, Qwen/Jina, sampling/thinking и бюджеты не изменены.
Содержание наблюдения требуется обосновывать одинаково во всех режимах.

Новая revision: `2026-11-16-archive-verification-v1`. `contract.json` сохраняет
`prompt_revision`, новый `prompt_sha256` и implementation hashes; requests,
responses и messages сохраняются прежним способом. Это новый execution contract:
никакого resume из старых runs или смешения старого A с новым C. CPU-проверки не
доказывают, что Qwen стал чаще или уместнее искать либо точнее отвечать.

## Подготовка WSL: две независимые review-страницы, только CPU

Все команды выполняются в foreground с видимым stderr и `tee`. Используются
существующие среды, RGB и annotations; alignment, модели и скачивание не запускаются.
Если output уже существует, выберите новый номер; не удаляйте прежний каталог.

Сначала сохранить доступ к исходным 16 решениям на **оригинальном manifest**.
`review-page` проверяет candidate SHA handoff; здесь нет переноса на другую revision
и автоматического approval. Архив V3.1 соответствует исходной версии pilot.

```bash
cd /home/sasha/Facade_Change_Benchmark &&
git pull --ff-only &&
set -o pipefail &&
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -u -m facade_change.agentic www review-page \
  --candidates runs/2026-11-15-facade-qa-candidates-001/candidates.json \
  --selection runs/2026-11-15-www-handoff-001/selection.private.json \
  --handoff runs/2026-11-15-www-handoff-001/handoff.private.json \
  --archive-corpus runs/2026-11-15-www-archive-v31-001/corpus.json \
  --out runs/2026-11-16-www-review-original-16-002 \
  2>&1 | tee runs/2026-11-16-www-review-original-16-002.console.log
```

Затем импортировать фактический **public package JSON V4 с доступными assets**.
Путь V4 неизвестен в Cloud, поэтому команда запрашивает его у владельца после
возвращения в WSL. Freeze берётся из проверенного корпуса, а не придумывается:
новые all-candidates используют именно этот knowledge_cutoff. `available_at`
локальных фото консервативно задан тем же freeze; это новая retrospective revision,
не изменение прежней доступности/дат в pilot. V4 нельзя переименованием выдать за V3.1
или заменить его корпус внутри старого execution contract.

```bash
cd /home/sasha/Facade_Change_Benchmark &&
set -o pipefail &&
read -r -p 'Абсолютный путь к public package JSON V4: ' WWW_V4_PACKAGE &&
{
  /home/sasha/miniconda3/envs/change-agentic-rag/bin/python -u -m facade_change.agentic www archive-import \
    --package "$WWW_V4_PACKAGE" --out runs/2026-11-16-www-archive-v4-002 &&
  WWW_V4_CUTOFF=$(/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["knowledge_cutoff"])' \
    runs/2026-11-16-www-archive-v4-002/corpus.json) &&
  /home/sasha/miniconda3/envs/change-agentic-rag/bin/python -u -m facade_change.agentic qa generate \
    --histories /home/sasha/Facade_Change_Benchmark/data/histories \
    --annotation-targets runs/2026-10-25-history-targets-001/index.json \
    --knowledge-cutoff "$WWW_V4_CUTOFF" --task-mode retrospective --available-at "$WWW_V4_CUTOFF" \
    --splits train val test --out runs/2026-11-16-www-candidates-all-v4-002 &&
  /home/sasha/miniconda3/envs/change-agentic-rag/bin/python -u -m facade_change.agentic www review-package \
    --candidates runs/2026-11-16-www-candidates-all-v4-002/candidates.json \
    --histories /home/sasha/Facade_Change_Benchmark/data/histories \
    --archive-corpus runs/2026-11-16-www-archive-v4-002/corpus.json \
    --exposed-questions runs/2026-11-16-www-public-16-001/questions.json \
    --out runs/2026-11-16-www-review-all-v4-002
} 2>&1 | tee runs/2026-11-16-www-prepare-all-v4-002.console.log
```

Если all-candidates **уже существует** на той же histories/annotation revision,
со всеми train/val/test и тем же V4 freeze/availability, его можно передать в
review-package напрямую, пропустив generate. Не меняйте его JSON вручную.
Сохранённый pilot handoff/selection **не передаются** новому review-package:
совпадающие question IDs не разрешают перенос labels между разными candidate bytes.
`--exposed-questions` читает публичный manifest именно выполненного 16-case run,
исключает его ID/перекрывающиеся предложения и сохраняет building exposure;
модельные ответы при новом отборе не читаются. Другие использованные manifests
передаются повторениями этого флага.

Открыть две отдельные страницы:

- `runs/2026-11-16-www-review-original-16-002/review.html`: все старые решения,
  включая case 03; их исходные файлы остаются неизменными.
- `runs/2026-11-16-www-review-all-v4-002/index.html`: новый отбор, inventory и
  identity template. Наличие страницы не означает наличие 100 случаев.

Цели остаются 40/30/20/10 **новых** вопросов, cap=12 на physical building.
Проверяйте `summary.json`: requested/selected/shortfall, target_shortfalls,
excluded/eligibility_reasons и physical_buildings; подробности — inventory и
`batch/selection.private.json`. Полный candidate manifest не делает все его
здания eligible. TRAIN/TEST с неизвестной экспозицией не становятся evaluation;
квоты не заполняются ценой снятия ограничений. Пары Voskova_10/Voskova_12 и
baskov_32/baskov_34 остаются pending, пока человек не подтвердил или не отклонил
identity с provenance. View restrictions, прежние splits/exposure сохраняются.

Для повторного экспорта после ручного identity/exposure audit используйте **новый**
output и существующие `--identity-map`/`--exposure-audit`; не редактируйте frozen
outputs. Подтверждённый mapping должен быть одинаковым в inventory, review-page,
plan/run и автоматически проверяться при score. Архивные correspondence proposals
по-прежнему отдельны от change-event GT. Здесь нет команд `qa review`/approval.

## Проверки и оставшаяся работа

Точечные CPU-команды (существующая Cloud-среда, без полного suite):
`python` ниже — `/workspace/work/facade-venv/bin/python`.

```bash
python -s -B -m unittest tests.test_2026_11_15_www tests.test_2026_11_15_www_curation tests.test_2026_11_16_www_review -v
python -s -B -m unittest tests.test_2026_11_15_www_smoke tests.test_2026_11_15_www_archive_review -v
python -u -s -B -m facade_change.agentic www demo --out /workspace/work/2026-11-16-www-archive-policy-cpu-001
git diff --check
```

Результат: **39 + 16 = 55 тестов OK**, demo **9/9 completed** с
`synthetic_fixture=true`, `models_executed=false`, `quality_claim=null`.
Проверен записанный новый prompt hash относительно SYSTEM из `28a8e6d`;
синтаксис обоих WSL-блоков проверен через `bash -n`.

Fixture-проверки охватывают local-only final без Jina, разрешённый retrieval,
предел rounds/model calls/tokens, одинаковые исходные сообщения A/B/C, новый
prompt contract и отказ несовместимому resume, C−A с errors/missing и reviewed
physical aliases. Сквозной separate-review тест создаёт новую candidate/archive
revision, сохраняет прежние байты/решения, передаёт exposure через публичный
manifest и сообщает недобор без повышения eligibility. Синтетические ответы
не проверяют способность Qwen понять инструкцию.

После возвращения WSL остаются: фактический экспорт и проверка недобора,
ручной identity/exposure/answer/box review, полный аудит 48 реальных строк по
raw messages/retrieval/exposure/citations/boxes/costs и проверка поведения новой
инструкции на настоящем Qwen. Для изолированной проверки prompt следует
переиспользовать исходные вопросы, V3.1 и настройки, выполнить все A/B/C под
новым контрактом в новом каталоге; переход одновременно на V4 и другую когорту
не является чистым сравнением prompt. Никакой такой GPU-запуск здесь не выполнен
и расширенный/ночной эксперимент не запланирован автоматически.

Scoring требует заранее frozen approved references и plan. Пилот с текстовыми
решениями не становится formal scored run задним числом. Без архива результатов
нельзя проверить фактические C−A значения, factual support или стоимость.
