# Controlled directed-difference fusion revision

This revision addresses the state-confidence failure documented in
[the diagnosis](2026-10-19-continuous-fusion-diagnosis.md). It adds an explicit
`--feature-design directed_delta` option to the existing cache-only runner.
The default `raw_pair` option retains the previous feature design for reproducibility.
Always use a new output directory; historical models and results are not overwritten.

## Fixed comparison

The five variants retain their names. Their meaning is recorded with
`feature_design=directed_delta` and schema `2026-10-19-directed-delta-events-v1`:

| Variant | Features |
| --- | --- |
| `semantic_delta` | Existing raw positive/negative DINO probability difference |
| `lpips_only` | Existing LPIPS binary-change baseline |
| `semantic` | Directed `p_to - p_from` |
| `semantic_lpips` | Directed difference, LPIPS |
| `semantic_lpips_msdzip` | Directed difference, LPIPS, MSDZip ABS |

For gain, d = p_after - p_before. For loss, d = p_before - p_after.
The event logit is `intercept + w_delta*d + w_lpips*LPIPS + w_msdzip*MSDZip`.
Independent state-confidence terms are absent. The existing Newton solver fits
the same weighted regularized objective; **each fitted w_delta must be strictly
positive**. A nonpositive solution blocks the run. The code does not flip its
sign, replace the fit, or search another regularization value.

With a positive semantic slope, gain wins if p_after > p_before, loss wins for
the opposite inequality, and exact equality abstains. Class overlaps remain
allowed. This is equivalent to selecting the higher of the two logits, without
losing direction through floating-point cancellation in their shared terms.
For nonzero differences the winning score is sigmoid(logit). LPIPS/MSDZip can
still amplify a small nonzero difference; the revision does not guarantee
immunity to nuisance variation or fix a class that DINO fails to recognize.

Pairs, ontology, geometric support, annotation policy, seed, event sampling and
population correction, facade/pair weights, standardization on fitting TRAIN,
L2=0.01, threshold target 5% binary TRAIN FPR, strict `>`, metrics, and all four
evaluation cohorts are retained. No new image inference or DINO/MSDZip training
is performed. The small fusion models are refitted on CPU.

All temporal TRAIN facades are taken from the prepared manifest, independently
of the larger training set of the frozen image models. TRAIN fusion-OOF excludes
one facade only from fitting/scaling/calibration of the new fusion, not from
the historical training exposure of DINO or MSDZip. VAL/TEST have informed this
revision's design, so the results remain exploratory.

## Mandatory parity control

Semantic-only sigmoid is a strictly increasing transform of positive difference.
Consequently its binary and directed AP and its independently calibrated decisions
must agree with `semantic_delta`. Both models are fitted/calibrated normally;
the control does not copy the baseline threshold or predictions.

The code checks every evaluated pair, including TRAIN fitting and each held-out
TRAIN facade. It compares binary and directed masks over the full common support
and binary/directed AP over the known annotation region. Learned delta scores
retain float64 to avoid collapsing distinct float32 differences after sigmoid.
AP equality tolerance is 1e-12; binary/event mask disagreement tolerance is zero.
The raw DINO difference is computed in the same float32 coordinate as before.

Results are saved to `2026-10-19_parity.csv`. Any mismatch blocks the run before
publishing a summary, preserving the failing check and fitted models for inspection.
A successful run records `parity_passed: true` in `run.json`. With the current
62-pair dataset and 30 TRAIN pairs, 92 checks are expected: 62 all-TRAIN-model
evaluations and 30 additional fusion-OOF evaluations. These counts are not hardcoded.

## Run locally

```bash
cd /home/sasha/Facade_Change_Benchmark &&
git fetch origin &&
git switch codex/2026-10-19-semantic-update-eval &&
git pull --ff-only &&
/home/sasha/miniconda3/envs/scd_bench/bin/python3.10 -u \
  -m scripts.real_eval.2026-10-19_continuous_semantic_updates \
  --feature-design directed_delta \
  --semantic-run runs/2026-10-18-semantic-run-001 \
  --change-run runs/2026-10-18-night-run-001 \
  --out runs/2026-10-19-directed-delta-eval-001
```

The output includes the usual summary, per-facade and per-pair CSVs, models,
thresholds, run/config/provenance JSON, maps, full TRAIN-OOF/VAL/TEST gallery and
HTML, plus the parity CSV. Provide `summary.csv`, `per_facade.csv`, `run.json`,
`resolved_config.json`, `2026-10-19_parity.csv`, and `models/` for comparison.
If blocked, provide `run.json` and the parity CSV if present. Runtime and memory
for this revision on the real data have not yet been measured.

## Validation

```bash
PYTHONPATH=tests python -m unittest \
  test_2026-10-19_directed_delta_updates \
  test_2026-10-19_continuous_semantic_updates \
  test_2026-10-19_semantic_updates \
  test_real_gt_calibration test_semantic_geometry -v
```

The new checks cover high-confidence persistence versus a large transition,
invariance to equal probability offsets, time reversal, rejection of nonpositive
semantic slopes, held-out facade/VAL/TEST exclusion, independently calibrated
mask/AP parity, deliberate parity failure detection, and a complete synthetic
run including model/map provenance and read-only inputs. Synthetic correctness
does not establish improvement on real facades.
