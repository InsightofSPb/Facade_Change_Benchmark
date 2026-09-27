# Baskov pair: first real SIFT visual review

## Evidence and scope

Run reported by the owner: `baskov-2016-2019-sift-002`, SIFT with MAGSAC,
reference image 55 and source image 54, `max_side=1024`, native-reference RANSAC
threshold 3 px. The run used `--allow-inferred-metadata`; the facade/view name
`baskov_34` and years 2016/2019 remain filename-derived metadata.

Evidence inspected: the owner's pasted diagnostics and five uploaded outputs:
`checkerboard_preview.jpg`, `overlay_preview.jpg`, and `detail_0.png` through
`detail_2.png`. Each detail contains reference, warped source, and absolute RGB
residual, in that order. The original photographs, full residual/support arrays,
`geometry.json`, and `run.json` were not provided for this review. Input hashes,
the original run's code fingerprint, and artifact hashes were not independently
verified. This is an exploratory visual review, not a benchmark result.

## Reported diagnostics

| Diagnostic | Owner-reported value |
| --- | --- |
| Matches / inliers | 249 / 108 |
| Median / p95 inlier reprojection error | 1.5517 / 2.7513 native reference px |
| Source / reference inlier hull fraction | 0.4746 / 0.4769 |
| Overlap | 286,974 px; 76.705% of reference support |
| Source-only / reference-only support | 75,363 / 87,153 px |
| Mean RGB absolute difference on overlap | 43.0969 on the 0–255 scale |
| Canvas | 645 × 838 px |

Both native-to-proxy transforms were identity: the inputs were already smaller
than the 1024 px limit. Increasing that limit would not change this run's proxy
images. Reprojection errors are measured on the estimator's selected inliers;
they are not independent estimates of dense registration accuracy.

## Visual observations

- The overall facade correspondence is plausible: the same windows, ornament,
  and wall sections occupy broadly corresponding locations.
- The overlay shows doubled architectural edges, particularly around the roof
  and projecting cornices. The current result does not establish sufficiently
  accurate local registration for small-defect evaluation.
- The reference and warped source show a broad brightness/colour difference.
  Residuals follow wall texture, ornament, and window boundaries. Their intensity
  cannot currently be attributed uniquely to material change.
- Windows contain strong differences. Reflections, interior visibility, and
  possible actual window changes need separate interpretation.
- The wall opening visible on the yellow wall in `detail_2.png` is present in
  both observations. Its whole residual region cannot be treated as newly
  appeared damage on the basis of these previews.
- Black checkerboard blocks at the outer boundary arise because the current
  preview alternates the two canvas images even where the selected observation
  has no support. The scorer independently uses the overlap mask. These blocks
  are a preview artefact, not evidence that an observed facade region vanished.

## Decision and next comparison

Keep this run as the first real SIFT diagnostic. Coarse visual registration is
plausible; acceptance for dense small-damage scoring remains pending. No damage
accuracy, temporal ground truth, or confirmed physical-change claim follows.

Next compare LoFTR on exactly the same ordered pair, native images, homography
estimator, and RANSAC threshold after checking the existing
`scd_bench` environment and locating a compatible local checkpoint. LoFTR still
feeds one global homography; improved matching does not guarantee that relief
or viewpoint-related local offsets disappear. Parallax is a possible explanation
for some residual geometry, not a confirmed diagnosis from these previews.

Judge the comparison on the same architectural locations; automatic detail crop
positions can change when the warp and overlap change. More inliers or a
lower global RGB difference alone do not demonstrate better preservation of
small changes. No full-corpus rescan or benchmark run is required for this step.
