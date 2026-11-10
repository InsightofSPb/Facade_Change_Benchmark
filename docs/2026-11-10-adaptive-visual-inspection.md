# Bounded adaptive visual inspection — 2026-11-10

Branch: `codex/2026-11-10-adaptive-visual-inspection`.
Pinned base: [`9aa159c0489e3a4ee15a089b1e214ade8d980fc3`](https://github.com/InsightofSPb/Facade_Change_Benchmark/commit/9aa159c0489e3a4ee15a089b1e214ade8d980fc3).
The remote `codex/2026-11-08-change-agentic-rag` was refreshed before editing and
still pointed to that base. No later or unrelated commits were adopted.

This implements inspection **inside supplied registered canonical regions**.
AnyChange proposals guide selection; they neither define GT nor constrain the
final box. It is not automatic whole-facade discovery. The hypothesis is that
choosing spatial evidence and an available historical date can improve visible
change localization relative to a deterministic inspection schedule under the
same permitted evidence universe and caps. No scientific improvement has been
established. Full H0/H1, C,V coding and compression are not prerequisites for this
engineering pilot, and it does not establish the dissertation's compression
contribution. Persistent identities, raw RGB, geometry/visibility separation,
human review and building-level evaluation remain required.

## Policies and frozen endpoint

| Policy | Evidence and selection |
|---|---|
| A `pair` | Current RGB pair, dates and geometric support; ordinary pair prompt/schema and support validator, with the inspection restriction to one region-grid box |
| B `fixed_history` | Pair plus up to two latest eligible historical observations, inspected over the whole supplied region |
| C `heuristic` | Pair, capped AnyChange proposal metadata/preview, date catalogue and allowed actions; inspect the pair at the highest-ranked proposal, then that box in the latest eligible history |
| D `agent` | Exactly C's inventory, catalogue, actions and caps; Qwen selects any valid region box/current pair or listed historical date, or stops |

C and D have identical frozen inventory hashes and action sets. A and B are
evidence-access baselines; all four are **not** an identical-evidence comparison.
SAM3 is disabled for both selectors. Its existing tool/adapter and AnyChange's
original frozen SAM ViT-H implementation remain unchanged. No optional SAM3
profile, retrieval, encoder benchmark, training, forecasting, ontology change,
token compression, Web access or new database is added.

All final answers use the same fresh Qwen judge prompt/schema/preprocessing and
the existing `validate_final` support checks. It sees the current pair/support
and actually acquired RGB evidence, without selector messages, proposals,
catalogue, suggestions or a tentative verdict. A has only the pair. Selection
stop is not a scientific prediction. Only `region_crop_pixels` is accepted for
the single final box; a change decision requires a nonempty box. A malformed,
degenerate or unsupported confident box is rejected, never repaired.

Primary endpoint: **D minus C in mean building event F1, K=1, IoU=0.25**.
For each building with at least one eligible reviewed positive event:

`F1 = 2*matched / (2*matched + unmatched_predictions + missed)`.

The macro membership comes from reviewed positive-event counts before considering
prediction success. Every approved case remains in the cohort. Stable cases in
those buildings contribute false alarms. Stable-only buildings are reported
separately. Counts come from the repaired QA evaluator, including physical-event
deduplication and one-to-one matching; inspection does not implement a second
matcher. K=3/5/10 never pad the one prediction. The public protocol remains
`supplied_region_qa`.

Secondary output includes IoU=0.5, event precision/recall, stable false alarms,
completed/abstained/missing/error counts, decision coverage, justified abstentions
and unsupported confident decisions on unassessable cases, per-building paired
differences, proposal coverage and recovery outside proposals. Proposal coverage
means a selected proposal box has IoU>=0.25 with a reviewed event representation;
the whole-region empty-map fallback is not a detection. This diagnostic uses
saved QA matches for recovered/missed counts, not another assignment algorithm.
No crop-level significance claims are made. Genuine reviewed temporal masks may
enable the evaluator's fixed-reference pixel diagnostics. Neither boxes nor
single-date semantic masks are converted into temporal masks.

## Inputs, preflight and curation

```text
python -m facade_change.agentic inspection preflight --questions PUBLIC --histories HISTORIES --config LOCAL_CONFIG --protocol PROTOCOL --out NEW_OUT
python -m facade_change.agentic inspection run       --questions PUBLIC --histories HISTORIES --config LOCAL_CONFIG --protocol PROTOCOL --out NEW_OUT
python -m facade_change.agentic inspection evaluate  --questions PUBLIC --answers PRIVATE --run RUN --out NEW_OUT
python -m facade_change.agentic inspection demo      --out NEW_OUT
```

These are new commands. `run` has no private answers argument. `evaluate` runs no
models and reuses `2026-11-10_qa_evaluation.evaluate` once per policy, including
immutable review/candidate/GT checks. Missing, malformed or hash-mismatched
individual predictions become per-case failures; cohort/privacy/GT and frozen
protocol/inventory integrity violations remain fatal. Exit code is 0 for
completed, 2 for ineligible, failed or completed_with_errors. stdout is one JSON
result; existing fingerprint progress goes to stderr.

Preflight loads no learned model or worker. It validates the actual local config,
strict protocol, manifests, native RGB, saved transforms, availability and exact
public-to-canonical joins. Both manifest SHAs are mandatory. It reports each
question's eligible history IDs/count, excluded IDs/reasons and failures. A
selected invalid question blocks inference; it is never silently removed.
Global configuration/manifest failures are saved in `validation_error.json`.
TEST/unassigned questions are rejected. Building/view/original-hash consistency
and image/crop-hash split separation are checked across manifest metadata without
opening held-out images or labels. Native files are verified only for the selected
pair and its permitted history. No alignment or feature dataset is regenerated.

`--histories` accepts the explicit canonical directory or its `manifest.json`.
The join uses region ID, observation ID, source/crop/support hashes, dates and
saved transforms. It does not infer identity from filenames or embeddings.
The existing prepared-pair and native-image validators are reused. Each case has
a frozen RGB/method inventory; the run uses its own RegionStore. Only frozen IDs
can be opened. Prior cases, another policy, predictions, hypotheses, human
confirmations and generic history-store results cannot enrich it.

Time has two independent meanings:

* Observation date orders images. Extra history must strictly precede reference.
* `knowledge_cutoff` bounds availability of the compared pair and every extra
  observation. The protocol's explicit `archive_available_at` must match the
  question export: an archive date, or the stated `observation_date` assumption.

Archive availability after source acquisition but before the question cutoff is
valid. The inspection case records both times. Legacy WWW's source-date cutoff
is unchanged. Unknown availability is never invented. This first date-choice
pilot requires at least two distinct eligible historical dates in addition to
the pair, normally at least four distinct dates. The actual counts are reported;
one historical image is not a date-selection experiment.

Reuse `qa generate/review/export`; no second review application is introduced.
Before approving the small cohort, review assessability against **the complete
permitted RGB history universe**, using the canonical history gallery/manifest
and the candidate RGB, not just the displayed pair. Record relevant history IDs
in the private review where appropriate. An insufficient-evidence pair may be
assessable with permitted history. Conversely, an earlier damage observation
cannot establish onset or a later transition unsupported by those observations.
The protocol declaration `assessability_reviewed_against` is the owner's explicit
attestation, not an automated visibility finding. Report A's restricted access
when interpreting abstentions.

The proposed development pilot is about 12 pairs across at least four buildings,
with reviewed positives, stable and genuinely unassessable cases where available.
Counts are not pre-existing dataset facts. First use a separately reviewed export
of exactly two cases with `policies=["heuristic","agent"]`; that profile enforces
two questions. The full four-policy export is reviewed separately. Model weights
and real experiments are executed by the owner after code review.

## Proposals, native crops and resources

AnyChange is prepared once per case or reused through the existing cache, whose
key binds raw RGB, geometric support, transforms, implementation, options and
checkpoint/source hashes. Missing `methods: anychange`, `source_root` or
`checkpoint_path` fails explicitly. There is no SSIM fallback or assumed weight
path. A valid empty map differs from a missing/failed map; only the former gets
the marked whole-region fallback.

Components are 8-connected within geometric support, without a size filter.
Order is descending component area, then ascending top, left, bottom, right and
component ID. At most three are exposed. The complete component list/count,
selected IDs, omitted areas and minimum envelope dimensions record small/thin
component loss due to the cap. These areas are not confidence scores. History
order is date descending, then observation ID descending, including deterministic
same-date ties; the catalogue exposes no historical RGB.

`inspect_region(observation_ids, bbox_xyxy)` accepts either the ordered two current
IDs or a single permitted history ID and a nonempty integer half-open box inside
the region. It maps the request through the saved `crop_to_source`, extracts an
enclosing crop from original RGB pixels and saves native offset, original shape
and hash, source polygon, alpha, geometric support and envelope-context mask.
Transforms `crop_local_to_observation_native` and `crop_local_to_region` identify
the child grid explicitly. A clipped native envelope is recorded. Singular
mappings, invalid boxes and excessive sizes fail without resizing. Envelope
context outside the polygon is not additional evaluated area.

Canonical 224x224 crops do not themselves prove higher-detail zoom. Original crop
dimensions and actual Qwen `processing.image_grid_thw` are saved; a tighter focus
is distinguished from finer source sampling. No generative upsampling is used.
Geometric support is not an occlusion label; unknown visibility alone does not
force abstention, and source-only support is not change.

The strict example is
[`configs/2026-11-10-inspection-protocol.example.json`](../configs/2026-11-10-inspection-protocol.example.json).
Its placeholders intentionally fail until bound to real inputs. C/D ceilings:
2 actions, 5 logical model requests, 8 images/prompt, 32 cumulative image
presentations, 16,384 input tokens/request and 65,536 cumulative, 1,024 output
tokens/request and 4,096 cumulative, 1 invalid-reply retry, no repeated action,
1,800 seconds/policy-case. No budget sweep is provided.

Configuration preflight reports exact conflicting values. The actual config
must permit these ceilings and a compatible worker timeout, sequential GPU
ownership, pinned Qwen revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, BF16,
SDPA, greedy decoding, local files only and the unchanged local image-processing
settings. It does not reconstruct or modify the owner's config. SAM3 remains at
its existing revision/checkpoint and is never called here. One RTX 4060 Ti 16 GB,
batch 1 and sequential workers are the intended local setup; 16k reliability and
multi-model residency are not established by these CPU checks.

Budget accounting includes planner, final judge, retries, repeated transcript
images, support/proposal previews and cache hits. Logical cached presentations
charge original usage; actual current worker costs and cache-source costs remain
separate in `calls`. Before a call, reserve two worst-case attempts for the
existing single OOM retry; before planning, additionally reserve a full fresh
judge. Remaining planner token ceilings may be reduced explicitly and recorded
in the trace; input is never truncated. Exact token overflow is rejected by the
unchanged real processor. Action prechecks validate image counts, native pixel
dimensions and remaining judge capacity before RGB delivery. At two actions,
proceed directly to the judge, without a planning turn just to stop.

Budget-limited selection is reported separately even if the reserved judge
completes. Exhausted generation/retry budgets and runtime/format errors are errors,
not successful abstentions. Malformed/truncated replies retain generation and
processing diagnostics. Unknown worker totals are null with observed lower bounds
and a reason. Shared preparation/candidate costs are recorded once per case;
incremental and total policy-case times are separate. Cold loading is summed only
from reported attempt measurements; exact warm inference time remains null when
the adapter reports only combined wall time. No invented timing zeros are used.

## Saved artifacts and CPU acceptance

The existing RunStore journal, source snapshots, immutable artifacts, RegionStore
and model/method caches are reused. A run saves resolved config/protocol, input
hashes, fixed public cohort and eligibility; each case saves `case.json`,
`inventory.json`, AnyChange maps, `proposals.json` and preview. Each policy saves
`contract.json`, planner/judge `messages.json`, `state.json`, `trace.jsonl`,
`evidence.json`, native `crops/*`, raw model responses, `final_judge_input.json`,
`result.json` and `costs.json`. Final judge input hashes bind model configuration,
request limits and image content independently of output-directory paths.

Each policy has a standard `*-predictions.json` wrapper binding the exact public
manifest. Evaluation saves the original QA outputs per policy and
`2026-11-10-comparison.json/.csv`. `report.html` links outcomes to compared dates,
crops, proposals, messages, usage and failure traces. Private evaluation output
stays separate from inference. Real RGB, private labels, local configs, model
weights and run directories are never committed.

The demo uses procedural 1024x768 RGB with one-pixel detail and nontrivial saved
perspective transforms, four dates, four buildings and five reviewed fixture
questions. Reviews are explicitly `synthetic_fixture_not_human`. Qwen and the
detector are scripted stubs available only through the demo/test seams; normal
`run` uses the existing real adapters.

Hand-checkable K=1/IoU=0.25 results:

| Policy | Eligible events | Matched | Missed | Unmatched predictions | Mean positive-building F1 |
|---|---:|---:|---:|---:|---:|
| C | 2 | 1 | 1 | 1 | (2/3 + 0)/2 = 1/3 |
| D | 2 | 2 | 0 | 0 | 1 |

D-C = 2/3. Every policy has 4 completed, 1 legitimate abstention, 0 errors.
One building is stable-only, one is unassessable; the stable case in a positive
building contributes C's false alarm. One event is recovered outside proposals.
These scripted numbers verify implementation, not scientific performance.

Focused Cloud acceptance commands from the repository root:

```bash
/workspace/work/facade-venv/bin/python -s -B -m unittest 'tests/test_2026-11-10_inspection.py' -v
/workspace/work/facade-venv/bin/python -s -B -m unittest tests.test_2026_11_08_agentic_reasoning tests.test_2026_11_08_agentic_tool_regressions tests.test_2026_11_10_facade_questions -v
/workspace/work/facade-venv/bin/python -s -B -m unittest tests.test_2026_11_08_agentic_cli -v
/workspace/work/facade-venv/bin/python -s -B -m unittest discover -s tests -p 'test_workflow.py' -v
/workspace/work/facade-venv/bin/python -s -B -m facade_change.agentic inspection --help
/workspace/work/facade-venv/bin/python -s -B -m facade_change.agentic inspection demo --out /workspace/work/2026-11-10-inspection-demo-002
git diff --check
```

Use the explicit dated test path: ordinary unittest filename discovery skips
hyphens. The existing `test_workflow.load_tests` hook also registers this suite
for normal full discovery. An initial `discover -p test_2026-11-10_inspection.py`
found zero tests; the explicit path executes the tests. No full repository suite,
learned models, weights download or real benchmark is part of this acceptance.
The first package-name invocation of `tests.test_workflow` produced one existing
fixture import error (`test_hypothesis_dataset`); use discovery for that module,
which puts the fixture directory on the import path. No fixture code was changed.

Cloud results for implementation commit
[`9d328d9`](https://github.com/InsightofSPb/Facade_Change_Benchmark/commit/9d328d9):

| Check | Result |
|---|---|
| Explicit dated inspection test file | 24 tests passed; no skips |
| Required reasoning/tool-regression/facade-question modules | 45 tests passed; no skips |
| Existing agentic CLI module | 7 tests passed; no skips |
| Workflow discovery, including its dated-suite hook | 46 tests passed, including the same 24 inspection tests; no skips |
| Inspection parent and all four subcommand help routes | Passed |
| CLI demo and stdout JSON | Completed with the expected counters |
| Demo HTML links | 210 analysis links and 8 evaluation links checked; no missing artifacts |
| `git diff --check` | Passed |

The isolated Cloud environment used the repository's pinned CPU requirements,
without Torch/Transformers or learned weights. The commands above cover 98
distinct tests; the dated suite is intentionally also checked through discovery.
The two initial invocation issues (zero tests from the hyphen filename pattern
and one existing fixture import error) are reported above, not counted as passes.

Changed files:

| File | Purpose |
|---|---|
| `facade_change/agentic/2026-11-10_inspection.py` | New CLI orchestration, RunStore artifacts, QA adaptation and static report |
| `facade_change/agentic/2026-11-10_inspection_data.py` | Strict protocol/preflight, frozen RGB joins, components and native crop tool |
| `facade_change/agentic/2026-11-10_inspection_policy.py` | Bounded selectors and fresh judge, accounting and traces |
| `facade_change/agentic/2026-11-10_inspection_demo.py` | Explicit procedural CPU fixture and stubs |
| `facade_change/agentic/cli.py` | Dispatch/help registration only |
| `tests/test_2026-11-10_inspection.py` | Focused acceptance tests |
| `tests/test_workflow.py` | Register the ISO-dated suite in the existing discovery hook |
| `configs/2026-11-10-inspection-protocol.example.json` | Strict dated protocol example with unresolved real-input placeholders |
| `docs/2026-11-10-adaptive-visual-inspection.md` | Protocol, validation and WSL handoff |
| `README.md` | Link to the new entry point and protocol |

## WSL worktree and CPU handoff

The branch is published separately. This setup preserves the WWW checkout and
does not force, reset or delete an existing worktree/local branch. If the branch
is already checked out elsewhere, Git refuses `worktree add`; inspect
`git worktree list` and use that worktree rather than forcing it.

```bash
FACADE_INSPECTION_BRANCH=codex/2026-11-10-adaptive-visual-inspection
git -C /home/sasha/Facade_Change_Benchmark fetch origin "$FACADE_INSPECTION_BRANCH"
git -C /home/sasha/Facade_Change_Benchmark ls-remote --exit-code --heads origin "refs/heads/$FACADE_INSPECTION_BRANCH"
if [ -e /home/sasha/Facade_Visual_Inspection ]; then
  test "$(git -C /home/sasha/Facade_Visual_Inspection rev-parse --show-toplevel)" = /home/sasha/Facade_Visual_Inspection || exit 1
  test "$(git -C /home/sasha/Facade_Visual_Inspection branch --show-current)" = "$FACADE_INSPECTION_BRANCH" || exit 1
elif git -C /home/sasha/Facade_Change_Benchmark show-ref --verify --quiet "refs/heads/$FACADE_INSPECTION_BRANCH"; then
  git -C /home/sasha/Facade_Change_Benchmark worktree add /home/sasha/Facade_Visual_Inspection "$FACADE_INSPECTION_BRANCH" || exit 1
else
  git -C /home/sasha/Facade_Change_Benchmark worktree add -b "$FACADE_INSPECTION_BRANCH" /home/sasha/Facade_Visual_Inspection "origin/$FACADE_INSPECTION_BRANCH" || exit 1
fi
cd /home/sasha/Facade_Visual_Inspection
git pull --ff-only origin "$FACADE_INSPECTION_BRANCH"
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -s -B -m facade_change.agentic inspection --help
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -s -B -m unittest 'tests/test_2026-11-10_inspection.py' -v
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -s -B -m unittest tests.test_2026_11_08_agentic_reasoning tests.test_2026_11_08_agentic_tool_regressions tests.test_2026_11_10_facade_questions tests.test_2026_11_08_agentic_cli -v
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -s -B -m facade_change.agentic inspection demo --out /home/sasha/Facade_Visual_Inspection/runs/2026-11-10-inspection-demo-001
```

Expected: completed demo, `analysis/report.html`, `analysis/*-predictions.json`,
`evaluation/{pair,fixed_history,heuristic,agent}/evaluation.json`, comparison JSON/
CSV and the hand counts above. Repeat with a new output suffix. These exact
help/demo arguments match the implemented parser; the WSL Python itself is not
present or executable in Cloud.

## Owner-run real smoke after code review

The recorded history location and SHA are checked below; a mismatch requires
explicit investigation and a reviewed protocol revision, not finding a different
latest dataset. The real local config was not read in Cloud. Its absence here,
missing local AnyChange options and absence of an approved real QA export are
unresolved local inputs, not demonstrated code failures.

```bash
cd /home/sasha/Facade_Visual_Inspection
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -s -B - <<'PY'
import hashlib
from pathlib import Path
p = Path('/home/sasha/Facade_Change_Benchmark/data/2026-11-04-histories-s192-001/manifest.json')
actual = hashlib.sha256(p.read_bytes()).hexdigest()
print(actual)
assert actual == 'dec521ea95cdb1447cd707ec31cc78c230864986a34c21c18c70f959415e3134', 'Inspect changed manifest; do not silently substitute another dataset'
PY
read -r -p 'Explicit archive availability (ISO date or observation_date): ' FACADE_INSPECTION_AVAILABLE
read -r -p 'Explicit knowledge cutoff (ISO date): ' FACADE_INSPECTION_CUTOFF
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -s -B -m facade_change.agentic qa generate \
  --histories /home/sasha/Facade_Change_Benchmark/data/2026-11-04-histories-s192-001 \
  --task-mode retrospective --knowledge-cutoff "$FACADE_INSPECTION_CUTOFF" \
  --available-at "$FACADE_INSPECTION_AVAILABLE" --splits val \
  --out /home/sasha/Facade_Visual_Inspection/runs/2026-11-10-inspection-candidates-001
cp -n /home/sasha/Facade_Visual_Inspection/runs/2026-11-10-inspection-candidates-001/review_template.json \
  /home/sasha/Facade_Visual_Inspection/runs/2026-11-10-inspection-decisions-001.private.json
```

Review the actual generated candidates and their permitted older RGB. Approve
exactly two suitable temporal questions in the copied decisions for the first
C/D smoke; leave all other candidates unapproved. Review labels, event IDs,
frames and assessability; machine proposals are not automatically approved. Then:

```bash
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -s -B -m facade_change.agentic qa review \
  --candidates /home/sasha/Facade_Visual_Inspection/runs/2026-11-10-inspection-candidates-001/candidates.json \
  --decisions /home/sasha/Facade_Visual_Inspection/runs/2026-11-10-inspection-decisions-001.private.json \
  --out /home/sasha/Facade_Visual_Inspection/runs/2026-11-10-inspection-review-001
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -s -B -m facade_change.agentic qa export \
  --review /home/sasha/Facade_Visual_Inspection/runs/2026-11-10-inspection-review-001/reviewed.json \
  --out /home/sasha/Facade_Visual_Inspection/runs/2026-11-10-inspection-export-001
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -s -B - <<'PY'
import hashlib, json
from pathlib import Path
repo = Path('/home/sasha/Facade_Visual_Inspection')
questions = repo / 'runs/2026-11-10-inspection-export-001/questions.json'
histories = Path('/home/sasha/Facade_Change_Benchmark/data/2026-11-04-histories-s192-001/manifest.json')
public = json.loads(questions.read_text())
assert len(public['questions']) == 2, 'Approve exactly two smoke cases'
available = {o['available_at'] for q in public['questions'] for o in q['observations']}
# Copy the explicit assumption actually used by qa.generate, including observation_date.
candidate = json.loads((repo / 'runs/2026-11-10-inspection-candidates-001/candidates.json').read_text())
protocol = json.loads((repo / 'configs/2026-11-10-inspection-protocol.example.json').read_text())
protocol.update(questions_sha256=hashlib.sha256(questions.read_bytes()).hexdigest(),
                history_manifest_sha256=hashlib.sha256(histories.read_bytes()).hexdigest(),
                archive_available_at=candidate['source']['available_at'], policies=['heuristic', 'agent'])
with (repo / 'configs/2026-11-10-inspection-smoke-001.local.json').open('x') as stream:
    json.dump(protocol, stream, indent=2)
    stream.write('\n')
PY
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -s -B -m facade_change.agentic inspection preflight \
  --questions /home/sasha/Facade_Visual_Inspection/runs/2026-11-10-inspection-export-001/questions.json \
  --histories /home/sasha/Facade_Change_Benchmark/data/2026-11-04-histories-s192-001 \
  --config /home/sasha/Facade_Change_Benchmark/configs/2026-11-08-agentic-wsl-16k-001.local.json \
  --protocol /home/sasha/Facade_Visual_Inspection/configs/2026-11-10-inspection-smoke-001.local.json \
  --out /home/sasha/Facade_Visual_Inspection/runs/2026-11-10-inspection-preflight-001
```

Preflight may report insufficient dates, missing native files, mismatched hashes
or missing/incompatible local model options. Resolve exactly those inputs and
rerun into a new directory. If a config revision is needed, preserve the original
local config and use the actual new config path consistently below; do not fill
in unknown weights or processing settings from this document. `run` also performs
preflight, so bypassing the separate command cannot bypass these checks.

After the owner reviews code and successful preflight, the real two-case commands
are (no private path is passed to inference):

```bash
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -s -B -m facade_change.agentic inspection run \
  --questions /home/sasha/Facade_Visual_Inspection/runs/2026-11-10-inspection-export-001/questions.json \
  --histories /home/sasha/Facade_Change_Benchmark/data/2026-11-04-histories-s192-001 \
  --config /home/sasha/Facade_Change_Benchmark/configs/2026-11-08-agentic-wsl-16k-001.local.json \
  --protocol /home/sasha/Facade_Visual_Inspection/configs/2026-11-10-inspection-smoke-001.local.json \
  --out /home/sasha/Facade_Visual_Inspection/runs/2026-11-10-inspection-smoke-001
/home/sasha/miniconda3/envs/change-agentic-rag/bin/python -s -B -m facade_change.agentic inspection evaluate \
  --questions /home/sasha/Facade_Visual_Inspection/runs/2026-11-10-inspection-export-001/questions.json \
  --answers /home/sasha/Facade_Visual_Inspection/runs/2026-11-10-inspection-export-001/answers.private.json \
  --run /home/sasha/Facade_Visual_Inspection/runs/2026-11-10-inspection-smoke-001 \
  --out /home/sasha/Facade_Visual_Inspection/runs/2026-11-10-inspection-smoke-evaluation-001
```

Stop and diagnose persistent format/runtime errors, missing native correspondence,
or lack of genuine historical choice. After a successful reviewed smoke, curate
the proposed small full cohort, export to a new directory, bind new input hashes
and all four policies in a new protocol, then repeat preflight/run/evaluate using
those actual paths. Retain adaptive selection only for a useful localization/
false-alarm tradeoff or comparable quality at lower actual cost relative to C.
A gain with worse coverage, more unsupported decisions or excessive cost is
insufficient. Passing the demo alone is not a reason to scale.
