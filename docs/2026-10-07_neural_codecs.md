# Original ArIB-BPS adapter and H0 training

`arib_bps_abs` and `arib_bps_mod256` add the original neural lossless **image**
codec to the same saved ten-building, sixty-pair comparison. Existing MSDZip
remains available. A neural video adapter has not been added: usable author code,
weights and an exact RGB path for NeuralLVC could not be verified.

Setup checks out [ZZ022/ArIB-BPS](https://github.com/ZZ022/ArIB-BPS) at
`bdc3c61d7cf3377dc72d411a32713f5f353d61e8` into `third_party/arib_bps`.
Its 26 Python/C++ sources are verified against
`third_party/arib_bps_provenance.json` before import or compilation. The original
network, configuration and entropy coder are used directly. Downloaded author
sources are excluded from this repository's Git history; setup reproduces them.
Compilation supplies the missing standard `cstdint` header through a compiler
flag and does not edit author files.

## Adaptation and provenance

The adapter compresses either the absolute or modulo-256 RGB difference. ArIB
does not condition its prediction on the reference image separately. Original
SIG and INS networks are trained with their original objectives and Adam, from
fresh weights, on unchanged, non-sham H0 train tiles. Checkpoint selection uses
H0 VAL theoretical cost averaged within each building, then equally across
buildings. No TEST RGB or damage labels enter training or checkpoint selection.

The default pilot uses 160 train and 16 VAL tiles of 32×32×3 bytes, three epochs,
batch one and learning rate 0.0002. These budgets are 491,520 and 49,152 RGB bytes
before repetition, close to the existing small MSDZip pilot budgets. Selection
round-robins building/scenario cells and ranks nonoverlapping tiles by SHA-256.
The same selected tiles are used for both representations. Unsupported pixels
and boundaries are zero-filled; loss includes the complete padded tile, and each
tile's support fraction is recorded. This is bounded facade-domain adaptation,
not a reproduction of the published ImageNet training or a performance claim.

Runs have new directories, immutable sampling plans, source/coder/checkpoint
hashes, environment records, training histories and a hashed summary. Loading
rejects edited configuration, artifacts or a different dataset fingerprint.
Scoring retains the fixed 32-pixel tile, 16-pixel stride and raw-cost score scale.

## Two explicit cost modes

| Mode | Score input | Reconstruction evidence |
| --- | --- | --- |
| `bitstream` (adapter default) | Actual complete author stream bytes, including header and bits-back initialization | Every tile is decoded and every RGB byte compared; mismatch fails the run |
| `theoretical` | Original `inference().sum()` variational bits per RGB byte | No stream measured during scoring; metadata explicitly says reconstruction was not checked |

Weights are not charged to the anomaly score. Overlapping independent tile
streams are not a whole-image compression rate. The common runner saves cost
mode and actual-byte statistics in `codec_stats.json`; theoretical costs never
appear as measured bytes.

The author's SIG decoder accumulates normalized bitplanes differently from the
encoder's `uint8 / 255`. A float32 difference changed posterior CDF entries and
corrupted bits-back restoration on a sparse tile. A reversible adapter wrapper
canonicalizes only decoder `sig.lvae._compress_qz` input as
`round(x * 255).to(uint8).float() / 255`. Encoder stream bytes, author source
files, layers and entropy coder are unchanged. This correction is recorded in
metadata and does not replace the mandatory exact check.

Exact encode/decode of a 32×32 tile with the original ImageNet32 configuration
took roughly **16 seconds on the validation CPU**. A 256×256 pair has 256 tile
origins at stride 16, about 70 minutes on that CPU. GPU timings and numerical
behavior must be checked at home. The recipe below uses theoretical scoring for
the exploratory comparison, preceded by actual bitstream controls. Use
`--cost-mode bitstream` to measure actual streams for the complete comparison.

## Home launch

Keep this as separate short commands. Setup clones `scd_bench` into the isolated
`facade-codecs` environment and verifies inherited dependencies; it does not
change the existing environment or download model weights. `conda`, Git and a
C++ compiler are required. Existing FFmpeg and RSCD setup remains as documented
in [the classical-codec delivery](2026-10-07_classical_codecs_rscd.md).

```bash
cd /home/sasha/Facade_Change_Benchmark
git pull --ff-only
bash scripts/setup_neural_codecs.sh
conda activate facade-codecs
bash scripts/setup_codecs.sh
NEURAL_RUN="$(date -u +%Y%m%dT%H%M%SZ)"

# Resolve the dataset from the existing local configuration.
DATASET_RUN="$(python -c 'from facade_change.benchmark_config import benchmark_arguments; print(benchmark_arguments({"benchmark_config":"configs/benchmark.local.json"})["dataset_run"])')"

python -u -B -m facade_change arib-train \
  --dataset-run "$DATASET_RUN" \
  --source-root third_party/arib_bps \
  --representations abs mod256 --author-config imagenet32_config \
  --device cuda:0 --epochs 3 --batch-size 1 \
  --max-train-patches 160 --max-val-patches 16 \
  --out "runs/${NEURAL_RUN}-arib-h0"
```

Then create a new comparison configuration with the original saved selection:

```bash
python scripts/configure_codecs.py \
  --reuse-run runs/2026-10-04-all-methods-003 \
  --training-run "runs/${NEURAL_RUN}-arib-h0" \
  --source-root third_party/arib_bps \
  --neural-device cuda:0 --cost-mode theoretical \
  --out "runs/${NEURAL_RUN}-codecs-rscd" \
  --config-out "configs/${NEURAL_RUN}-codecs-rscd.local.json"

python -u -B -m facade_change methods-check \
  --config "configs/${NEURAL_RUN}-codecs-rscd.local.json" \
  --methods arib_bps_abs arib_bps_mod256 --bitstream-check \
  --out "runs/${NEURAL_RUN}-arib-bitstream-check"
```

The controls cover zero, sparse paint residual and wraparound values. Continue
only after the command succeeds and its `report.json` confirms exact restoration
for both methods on the actual worker/device. A mismatch stops the command and
records a failed run; keep its report and worker logs for diagnosis.

Check the classical codecs and RSCD with the same configuration, then launch
the common comparison:

```bash
python -u -B -m facade_change methods-check \
  --config "configs/${NEURAL_RUN}-codecs-rscd.local.json" \
  --methods jpegls_mod256 h264_rgb rscd_cmu rscd_diff_cmu rscd_pscd \
  --max-val-bases 1 \
  --out "runs/${NEURAL_RUN}-classical-rscd-check"

python -u -B -m facade_change h0h1-benchmark \
  --config "configs/${NEURAL_RUN}-codecs-rscd.local.json"
```

All unchanged existing methods reuse verified maps and original thresholds. RSCD
is recalibrated with the finer fixed grid; new codecs use the identical saved
selection. Earlier outputs and configuration files are never overwritten. The
helper adds only representations present in the completed training run. Use a
different `--reuse-run` if the completed source run has another name.

## Verification scope

CPU validation reproduced the original sparse reconstruction failure, decoded
the same stream exactly with the wrapper, and verified that seeded encoder bytes
remained unchanged. Zero and random controls also restored exactly. A separate
tiny original-model H0 training fitted both representations on one train and one
VAL identity tile for one epoch. Six trained zero/sparse/random streams restored
exactly, and modified config, summary, checkpoint and dataset hashes were rejected.
The identity sample gives zero residuals in both representations; equal smoke
weights and costs are expected and do not establish detection quality. Isolated
workers also passed theoretical scoring with correct native support/metadata.
The full repository suite completed 314 tests with no failures or errors and one
optional author smoke skipped; that author smoke passed separately against the
verified sources. Full dataset training, GPU inference and detection-quality
evaluation remain home experiments.
