# Diagnosis of the continuous semantic fusion pilot

Analysis date: 2026-10-19. Source run: `2026-10-19-continuous-semantic-update-eval-001`,
completed, 62 temporal pairs, 4 temporal TRAIN facades, 264.93 seconds.
Source implementation: published commit `0a6ce162357b0ad4b090eaa5f5f064a52cba483d`.
All eight code SHA-256 values in the supplied resolved config match the inspected
local implementation. Summary metrics reconstruct exactly from the supplied
per-facade rows; no raw NPZ files were supplied for this diagnosis.

## Verified failure in the learned semantic score

For the fusion fold excluding `Ulitsa_Mira_15`, the fitted semantic gain score
before the direction-selection rule is:

`sigmoid(-5.251020148459506 + 0.5953001019882899 * p_before + 1.8948877855990707 * p_after)`.

The calibrated binary threshold is `0.05938853695988655`, with a strict `>` operator.
For gain, the chosen direction requires `p_after > p_before`; loss swaps the two
probabilities. Exact equality abstains. Both learned coefficients are positive.
Consequently, the score rewards confidence that the class is present on both dates.

Controlled inputs passed through the actual `score_pair` implementation, float32:

| p_before | p_after | Learned semantic score | Learned decision | Probability-difference decision |
| --- | --- | --- | --- | --- |
| 0.1 | 0.9 | 0.02971058 | no change | change |
| 0.999 | 1.0 | 0.05944450 | change | no change |
| 0.99 | 0.99 | 0 | no change | no change |
| 0.0 | 1.0 | 0.03369493 | no change | change |

The probability-difference threshold for the same fold is 0.5871707797050476.
These are diagnostic input probes, not observed image pixels or empirical event
probabilities. They prove a property of this fitted function, not the prevalence
of these exact inputs in the dataset.

Because both coefficients are positive and probabilities lie in [0,1], any
above-threshold gain must have p_before > 0.99731788 and p_after > 0.99915738
(necessary individual bounds). The mathematical score upper bound is 0.05947779,
only slightly above the chosen threshold. The direction rule imposes an additional
ordering constraint. Loss has the corresponding swapped bounds.

This is consistent with the empirical failure: over the five held-out Mira pairs,
the semantic variant has **0 TP, 16,756 FP, 92,083 FN** for binary change. Every
accepted binary prediction is false according to the evaluated annotation-change
labels. The raw probability-difference baseline has facade-mean F1 **45.61%**;
the learned semantic variant has **0%**. These counts pool pixel observations over
the five pairs for diagnosis; they are not independent samples for inference.

The full-TRAIN semantic model also contains a state-confidence component:
its raw coefficients are -0.44517 and +2.96225, not equal and opposite. At that
model's threshold the examples above behave differently, so the Mira degeneracy
must not be generalized to every fold. The broader issue is that the design
permits state confidence to replace change magnitude as the principal signal.

## Image evidence and its limits

The supplied HTML/gallery is from the earlier **hard-AND** pilot. It reuses the
same frozen semantic states and aligned RGB, so it can diagnose those inputs.
Its prediction/error overlays must not be described as the new continuous-model
outputs. The images are reduced to at most 384 pixels per side: precise alignment
or annotation-boundary errors cannot be quantified from these previews.

- `Voskova_10`, 2019 to 2021: substantial surface changes are visible alongside
  strong exposure/color changes. The semantic masks identify parts of the changed
  surface but also vary elsewhere. This is a useful positive diagnostic case,
  not evidence that nuisance variation has been separated from physical change.
- `marata_44`, 2019 to 2022: broad exposed/patchy areas look similar across the
  inspected dates, while the predicted class masks vary substantially around
  windows and damaged surfaces. Browser/panorama text overlays are visible.
  Photometric sensitivity, screenshot overlays and local alignment are plausible
  contributors; their causal contributions remain unmeasured.
- `manezhnyi_7`, 2009 to 2011 and 2013 to 2019: semantic overlays emphasize
  cornices and ornament; the displayed annotation changes are small. The numeric
  class audit below establishes which predicted events are actually false.

## Class-level diagnostic counts

All counts below are directed pixel/class/gain-or-loss observations over the
relevant facade's pairs, not unique spatial pixels or facade-macro metrics.

- Manezhny, continuous `semantic`: **1,064 of 1,846 event FP (57.64%)** are
  `ORNAMENT_INTACT`, with no annotated ornament events. Its 3,626 positive
  `DELAMINATION` event observations are all missed. Adding LPIPS suppresses the
  ornament errors, but the combined variant has **0 binary TP** across all five
  pairs; it therefore does not recover the missing signal.
- Marata, `semantic_lpips`: **3,215 of 7,041 event FP (45.66%)** are
  `TEXT_OR_IMAGES`, with no annotated events in that class. Screenshot labels
  are a plausible explanation to inspect in raw maps, not a proven attribution
  of those 3,215 observations. The variant detects only 117 of 30,877 positive
  directed event observations.
- Voskova, `semantic_lpips`: the binary facade-mean F1 is **60.84%**. Directed
  event counts are **43,927 TP / 28,505 FP / 81,889 FN**, showing that locating a
  changed region does not establish its correct class and direction. All 5,273
  annotated crack event observations are missed at the chosen threshold.

These findings do not justify removing difficult classes from the benchmark or
relabelling physical events from model predictions. Unknown annotation remains
unknown; GT here is a difference in annotation membership.

## Next targeted comparison

The next justified code experiment would replace the two independent semantic
features `(p_from, p_to)` with a directed difference `p_to - p_from`, requiring a
positive semantic slope, while retaining the existing LPIPS/MSDZip feature
additions, source pairs, labels, weighting, regularization and TRAIN-only threshold
calibration. It would isolate the demonstrated confidence-versus-change failure
without changing the class set or selecting favorable pairs.

For the semantic-only variant, a positive affine transform followed by sigmoid
is strictly monotone in the difference. Its ranking and TRAIN-quantile decisions
should therefore agree with the raw difference baseline, apart from numerical
ties; that parity is a required control. No accuracy improvement is claimed
before running the comparison. Remaining image-level semantic errors would still
need attention. VAL/TEST have already informed this diagnosis, so any subsequent
result remains exploratory.

This note records an analysis, not a new model run or a modified historical result.
