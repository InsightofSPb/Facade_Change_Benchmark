# Resumable temporal change experiment

The `temporal-experiment` command executes the requested TRAIN/VAL comparison:
preflight, real inference smoke checks, MSDZip H0 training, pair comparisons,
causal state-memory ablations, and reports rebuilt from saved maps. It has no
72-hour limit. It does not train the semantic DINO head or load TEST images.

This implements the minimal history experiment requested on 14 October. It is
an exploratory comparison of change scores, **not** the master map's full paid
H0/H1 codec, a damage diagnosis, or evidence that history improves accuracy.

## Launch on the owner's machine

The paths below are the existing dataset and LoFTR batch supplied in the task.
A batch containing rejected alignments remains usable: accepted children are
used and rejected/missing observations remain in the coverage report.

```bash
cd /home/sasha/Facade_Change_Benchmark
git pull --ff-only
/home/sasha/miniconda3/envs/scd_bench/bin/python -u -s -m facade_change temporal-experiment \
  --dataset /home/sasha/Facade_Change_Benchmark/runs/20261011T175248818684449-dataset-402/dataset.json \
  --alignment-runs /home/sasha/Facade_Change_Benchmark/runs/20261013T202601703319431-val-loftr \
  --semantic-run auto \
  --methods-config auto \
  --device cuda:0 \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-10-14-temporal-experiment-001 \
  --detach
```

`auto` reuses `configs/methods.local.json`, falling back to the existing example
paths only when no local config exists. Semantic `auto` selects the latest
completed, non-smoke semantic training run in this checkout's `runs/` directory.
The exact selected run, worker interpreters and options are saved in
`request.json`, `resolved_config.json`, and `method_provenance/`. To select a
particular trained head, supply its directory with `--semantic-run /absolute/path`.
No model weights or dependencies are downloaded by this command.

The launcher prints its PID and log path. Follow the log with:

```bash
tail -f /home/sasha/Facade_Change_Benchmark/runs/2026-10-14-temporal-experiment-001.log
```

The worker runs independently of this terminal. Keep the machine awake.
After interruption or reboot, use the **same output path**:

```bash
cd /home/sasha/Facade_Change_Benchmark
/home/sasha/miniconda3/envs/scd_bench/bin/python -u -s -m facade_change temporal-experiment \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-10-14-temporal-experiment-001 \
  --resume --detach
```

For preliminary checks, use a fresh output path and add `--preflight-only` or
`--smoke-only`; continue that exact run with `--resume`. Omit `--detach` for
foreground execution. Missing model resources are reported per method and do
not destroy independent results. A common data/split/geometry error stops the
stages that require those inputs. Read `method_status.json`: a completed process
with unavailable methods is not a complete nine-method scientific comparison.

If an existing `prepare-history` run should be reused, supply
`--history-run /absolute/path` along with `--dataset`. Otherwise the command
creates deterministic five-frame clean/event histories under `prepared_history/`.
`--crops-per-building N` increases the number of selected native crops.
Multiple accepted alignment batches can follow `--alignment-runs`.

### Corrected MSDZip smoke and live progress

The initial implementation evaluated tiny smoke checkpoints (two target bytes
per forward) on full crops. One fully supported 512×512 pair therefore required
393,216 grouped forwards, before the additional history comparisons. The corrected
smoke uses at most 16×16 **native pixels** for MSDZip only: 384 forwards for the
equivalent fully supported pair. A deterministic maximum-common-geometric-support
window is shared across all five synthetic dates; the real pair has its own
window. No image resizing or annotation-based selection occurs. Coordinates and
original/scored shapes are saved in `smoke/msdzip_*.json` under `smoke_inputs`.
Other methods and the full experiment retain their original inputs.

The main log now announces preparation, source verification, method initialization,
individual smoke pairs/history frames, full training and reporting. Each MSDZip
inference also writes start, periodic byte/group counts, and completion to its
`logs/facade-worker-msdzip_*.log`. Updates are emitted at most once per five seconds
between start/completion, after a batch finishes; there is no background heartbeat.

If the original `...-001` run is still in its smoke stage, stop its recorded
coordinator PID with SIGTERM and confirm that it exits before updating the checkout.
This also closes its owned worker. Keep the old run directory. Because resume
verifies source hashes, start the corrected code in a **new** output directory,
reusing the completed synthetic preparation and existing alignment results:

```bash
cd /home/sasha/Facade_Change_Benchmark &&
git pull --ff-only &&
/home/sasha/miniconda3/envs/scd_bench/bin/python -u -s -m facade_change temporal-experiment \
  --config /home/sasha/Facade_Change_Benchmark/runs/2026-10-14-temporal-experiment-001/request.json \
  --history-run /home/sasha/Facade_Change_Benchmark/runs/2026-10-14-temporal-experiment-001/prepared_history \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-10-14-temporal-experiment-002 \
  --detach
```

This still launches the full experiment after corrected smoke checks. Add
`--smoke-only` to stop after the checks; subsequently use the same `...-002`
output with `--resume --detach` to continue. The smoke checks are repeated under
the corrected source version; `request.json` preserves the previous dataset,
alignment paths, selected semantic head and method settings. Previously saved
research artifacts are not modified.

## Methods and targets

| Method | Scoring and provenance |
|---|---|
| RGB-DIFF, SSIM | Existing native RGB implementations |
| LPIPS, DINOv2 | Existing frozen author-backed adapters |
| semantic DINO | Verified frozen backbone and saved sigmoid multilabel head; `max_c abs(p_source - p_reference)` |
| AnyChange, GeoSCD | Existing author-native binary predictions, no training |
| MSDZip ABS, MOD256 | Matched H0 training data, sampling seed and architecture; separate learned weights |

Semantic outputs preserve signed class-probability changes, separate uncertainty
maps, and optionally probabilities. The checkpoint's actual class/channel mapping
is validated; legacy 11-class and newer ontology projections are not silently
interchanged. Uncertainty never removes pixels from detection evaluation.
Checkpoint training overlap and unknown facade identities are reported explicitly.

All methods compare **earliest** and **previous usable observation**. DINOv2,
semantic DINO, MSDZip ABS and MOD256 additionally run **history**. Earliest-state
labels and adjacent-event labels remain separate evaluation tasks.

History stores actual prior RGB observations with masks describing which pixels
belong to the current local episode. It scores before updating memory; a detected
change clears older episode exemplars locally, including for repairs. The current
observation is then admitted. First observations receive initialization rather
than a change label. The default is three exemplars per pixel; sparse visibility
can retain more than three full images, because images are not stitched into
artificial RGB mosaics. Frame states are saved with each history result.

The memory reset threshold is fixed from previous-frame synthetic VAL H0 at 1%
facade-macro FPR. Detection thresholds for the resulting history scores are then
calibrated separately. This is a fixed causal policy, not a learned memory model.
The gallery's previous-image panel provides context; `reference_frame` identifies
the actual older reference selected per pixel by history.

## Calibration and scientific limits

MSDZip uses clean-arm adjacent **observed RGB** pairs: no new structural event,
with exposure, white balance, shadows and other generated observation changes.
Existing damage in the base photograph is allowed. Only TRAIN fits weights; VAL
selects the checkpoint. Clean-arm labels never enter the scorer or memory update.

Primary comparison: Recall at thresholds calibrated on synthetic VAL clean-arm
H0 for FPR <=1% and <=5%. 'Clean arm' means no new structural event and includes
all nuisance conditions. Counts are aggregated within each facade and then across
equally weighted facades. AnyChange/GeoSCD retain their native predictions and
report their actual Recall/FPR. VAL numbers remain exploratory calibration-set
results; no independent TEST claim or confidence interval is made here.

History versus previous uses identical adjacent-event labels/support and matched
completed frames. Real data receive prediction galleries and alignment/coverage
records, not temporal accuracy numbers from unreviewed semantic-mask differences.
Real crop selection uses geometry, not COCO transition labels. Missing/rejected
observations and ambiguous same-year chronology remain explicit exclusions.

Optional JSON settings include `legacy_checkpoints: {"abs": "/path/to/old.pt",
"mod256": "/path/to/old.pt"}`. These create additional legacy variants. Their
old training provenance and current VAL overlap are disclosed. Old versus new
weights change training data/adaptation as well as weights; this is not an
architecture-only ablation. A JSON config can also set `training` options accepted
by `msdzip-train` (epochs, byte budgets, architecture, learning rate and checkpoint
cadence), `method_options`, `memory_exemplars`, and `memory_update_fpr`.

## Saving and rebuilding

- Each unit has an atomic NPZ archive followed by an atomic JSON completion marker.
  An interrupted temporary file is never treated as complete.
- Resume checks config, input bytes, model/source hashes, worker environment and
  committed artifact hashes. Incompatible changes require a new output directory.
- History archives include post-frame memory and link to the preceding committed
  state. The unfinished frame is retried; completed frames are not re-applied.
- MSDZip checkpoints preserve model, optimizer, position, RNG and best-VAL state.
  `checkpoint_every_steps` defaults to 100; completed representations are reused.
- `events.jsonl` records attempts and failures. An interrupted trailing fragment is
  preserved separately before recovery. Concurrent runners cannot own one output.

Rebuild all tables and the standalone HTML gallery without models or source images:

```bash
cd /home/sasha/Facade_Change_Benchmark
/home/sasha/miniconda3/envs/scd_bench/bin/python -s -m facade_change temporal-experiment \
  --out /home/sasha/Facade_Change_Benchmark/runs/2026-10-14-temporal-experiment-001 \
  --report-only
```

Main outputs: `2026-10-14_temporal_summary.json`,
`2026-10-14_temporal_thresholds.json`, `2026-10-14_temporal_facades.csv`,
`2026-10-14_temporal_events.csv`, `2026-10-14_temporal_nuisances.csv`,
`2026-10-14_temporal_history_comparison.csv`, and
`2026-10-14_temporal_gallery.html`. `preflight.json` retains input/geometry coverage;
`method_status.json` and `logs/` retain per-method outcomes.

## Verification

```bash
python -s -m unittest discover -s tests -v
python -s -m unittest \
  'tests/test_msdzip.py' \
  'tests/test_semantic_dino.py' \
  'tests/test_temporal_inputs.py' \
  'tests/test_temporal_history.py' \
  'tests/test_temporal_reports.py' \
  'tests/test_temporal_experiment.py' -v
```

The second command is required because standard unittest discovery skips
hyphenated filenames. CPU fixtures check computation, persistence and recovery;
they do not establish real-facade accuracy or validate the owner's GPU environment.
