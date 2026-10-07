# Classical codecs and RSCD calibration, 2026-10-07

This delivery continues the existing frozen comparison: **10 crops from distinct
buildings, 60 pairs, 3 validation and 7 test buildings**. It introduces no new
crops, split, training or augmentation. NCD is deferred. ArIB-BPS and the neural
video codec belong to the next delivery.

## Implemented methods

| Method | Input | Charged cost |
| --- | --- | --- |
| `jpegls_mod256` | `(B-A) mod 256` RGB residual | Complete JPEG-LS stream, including headers |
| `jpegls_abs` | Absolute RGB residual; optional diagnostic | Complete JPEG-LS stream, including headers |
| `h264_rgb` | Original RGB frames A then B | P-frame packet with headers; reference I packet and full stream recorded separately |

Each encoded tile is decoded and compared byte-for-byte. A mismatch fails the
run. H.264 uses `libx264rgb`, QP 0, medium preset, no YUV conversion, no B frames,
one reference, and disabled scene cuts/lookahead. FFprobe must report exactly
one key packet and one non-key packet; their sizes must sum to the stream size.
FFmpeg/FFprobe binary hashes, versions and encoder options are recorded.

For JPEG-LS `mod256`, known A and decoded residual R reconstruct B exactly via
`(A+R) mod 256`. `abs` loses the sign and does not fully encode B given A.
For H.264 the P packet is conditional on the **decoded** I reference and its
state. It is not the standalone stream size or a claim that the I frame is free.

Local scoring uses 32x32 RGB tiles with stride 16, the existing top-left grid
and zero padding at incomplete edges. Only base geometric support enters the
scorer; unsupported input pixels are zeroed. Cost is divided by the full padded
tile's RGB-byte count. Raw bits per byte are averaged across covering tiles;
`1-exp(-bits_per_byte/8)` maps that average to the common score scale. Unsupported
outputs are NaN. There is no per-image min/max normalization or oracle mask.
Summed costs over overlapping independent tiles are **not** a whole-image
compression rate.

## RSCD: corrected calibration, unresolved transfer quality

The cached probability maps differ. A coarse grid `0, 0.01, ..., 1` cannot
resolve much of their probability distribution, and previously selected zero,
producing the same all-positive binary decision across checkpoints.

The new fixed RSCD grid retains every old threshold and adds 769 logarithmic
values from the minimum normal float32 value to 0.5. It is specified independently
of labels and TEST. Selection still uses building-macro H1 F1, then H0 FPR,
then the highest tied threshold, on VAL only. Other methods retain their old
grid. The selected threshold is frozen before TEST.

Review of the author code confirms RGB/255 with `wo_norm`, change channel 1 and
native last-axis argmax. No preprocessing or output-channel defect was found.
The adapter's previously disclosed patch-size padding remains unchanged.

Validation-only recalibration of the existing saved maps still showed poor
CMU/Diff-CMU ranking and high false-positive rates. **A finer threshold grid
repairs numerical calibration; it does not establish good pretrained RSCD
transfer to facade defects.** Fine-tuning or a separate transfer investigation
is subsequent work.

`methods-check` saves probability min/quantiles/max, distinct-value counts,
positive fraction at zero, author native masks, and A=A identity controls for
one selected VAL crop. It reads no TEST RGB. Run this on the owner's machine
to check actual checkpoints and CUDA behavior; Cloud unit tests do not establish
real-model accuracy.

## Preservation of old results

`recompute_methods` explicitly excludes the three RSCD methods from reuse after
hash validation of the complete source run. The remaining methods preserve
maps, predictions, original thresholds and recorded inference times. New methods
use the exact saved selection. Older run directories are never modified.
Codec costs live in hashed `codec_stats.json` and remain reusable in later runs.

## Commands on the owner's machine

Use the existing environment; this stage installs no Python packages or weights.
FFmpeg must include `jpegls` and `libx264rgb`. If missing, the preflight gives the
system installation command. RSCD paths and workers come from the existing
`configs/benchmark.local.json` and its methods configuration.

```bash
cd /home/sasha/Facade_Change_Benchmark
git pull --ff-only
conda activate scd_bench
CODEC_RUN="$(date -u +%Y%m%dT%H%M%SZ)"
bash scripts/setup_codecs.sh

python scripts/configure_codecs.py \
  --reuse-run runs/2026-10-04-all-methods-003 \
  --out "runs/${CODEC_RUN}-classical-rscd" \
  --config-out "configs/${CODEC_RUN}-classical-rscd.local.json"

python -u -B -m facade_change methods-check \
  --config "configs/${CODEC_RUN}-classical-rscd.local.json" \
  --methods jpegls_mod256 h264_rgb \
  --out "runs/${CODEC_RUN}-classical-check"

python -u -B -m facade_change methods-check \
  --config "configs/${CODEC_RUN}-classical-rscd.local.json" \
  --methods rscd_cmu rscd_diff_cmu rscd_pscd \
  --max-val-bases 1 \
  --out "runs/${CODEC_RUN}-rscd-check"

python -u -B -m facade_change h0h1-benchmark \
  --config "configs/${CODEC_RUN}-classical-rscd.local.json"
```

If the completed all-method run has another name, change only `--reuse-run`.
Omitting that option selects the most complete completed comparison, then the
latest, and reports the chosen path. It must contain all three RSCD methods and
exactly the existing 10 crops/60 pairs. An incomplete or corrupted source is
rejected. Verify the printed `reuse_run` before the benchmark.

The helper uses the active Python for classical workers. Override it with
`--worker-python /absolute/path/to/python` if needed. Existing RSCD worker
Python paths are preserved. Configuration files and outputs must be new paths.

## Outputs and checks

- Preflight/diagnostics: `run.json`, `report.json`, probability `.npy` files,
  RSCD `*-native.png`, worker logs.
- Comparison: existing `summary.txt/json`, `metrics.csv/json`, `gallery.html`,
  live progress, frozen thresholds, plus `codec_stats.json` with actual charged,
  reference-I and full-stream bytes and exact-roundtrip status.

CPU verification includes real FFmpeg lossless checks, a thin common-runner
fixture with real workers, cache/hash preservation, and focused RSCD calibration
and loader contracts. This verifies the software path, not scientific superiority.
All 289 repository tests completed successfully with no skips using an explicit
unittest result record. The diagnostic CLI passed six synthetic codec controls
with exact RGB reconstruction. No full real-data benchmark, training or real GPU
inference was launched in Cloud.
