# Directed-difference fusion: verified exploratory results

Run: `2026-10-19-directed-delta-eval-002`. Published source commit: `78bb6a4cbe193d3be6dc7a07e1b750d8f3974af1`.
Archive: `2026-10-19-directed-delta-results-002.tar.gz`; SHA-256: `b28c2cf876504348f195a73f815c31a559b590ddbfa8379d158a00968f6567ce`.

## Evidence audit

The run completed in 282.41 seconds on 62 temporal pairs. All 92 parity checks passed: binary and event masks match exactly; binary AP differences are zero; the largest absolute directed AP difference is 1.1255441023649837e-12 (declared absolute tolerance 1e-9). All eight code hashes match the inspected source. All five fitted folds have positive semantic slopes.
The 776 summary rows were reconstructed from 3,104 facade rows for precision, recall, F1, IoU, FPR and AP; maximum floating-point discrepancy was 1.12e-16. Both unchanged baselines match the previous raw-pair run exactly. Dataset membership, ontology, frozen models, sampling, seed, L2 and calibration target also match. The archive does not contain per-pair metric rows or pixel maps, so this audit does not independently recompute predictions from pixels.

## Protocol

Temporal TRAIN contains 4 facades / 30 pairs (29 positive); VAL 5 / 21 (20 positive); TEST 3 / 11 (11 positive). These are cached temporal subsets, not the full DINO training dataset. TRAIN fusion-OOF excludes each facade only from the new fusion fit, scaling and threshold calibration; frozen image models may have seen it.
All tables use all_frames. Metrics average pairs within each facade and then equally average facades, excluding undefined values. AP uses positive pairs only. Directed metrics count pixel/class/direction events inside each pair; they are not class-macro scores. Percentages below are on a 0–100 scale; deltas are percentage points. One threshold per variant is fitted to 5% binary FPR on fitting TRAIN facades. Reported held-out FPR is measured and is not constrained to 5%.
GT encodes changes in annotation membership, not independently verified physical events. A class loss is not proof of repair. TEST had already informed design decisions; all comparisons remain exploratory.

## Binary change detection

| Cohort | Variant | F1 % | FPR % | AP % | Precision % | Recall % |
| --- | --- | --- | --- | --- | --- | --- |
| train_fit | LPIPS | 15.897 | 5.000 | 27.316 | 34.743 | 13.246 |
| train_fit | DINO difference | 27.667 | 5.000 | 38.514 | 42.777 | 25.246 |
| train_fit | Difference + LPIPS | 26.940 | 5.000 | 38.872 | 43.830 | 24.387 |
| train_fit | Difference + LPIPS + MSDZip | 26.945 | 5.000 | 38.882 | 43.885 | 24.370 |
| train_fusion_oof | LPIPS | 17.286 | 6.077 | 27.316 | 33.550 | 15.493 |
| train_fusion_oof | DINO difference | 26.933 | 6.385 | 38.514 | 41.696 | 25.464 |
| train_fusion_oof | Difference + LPIPS | 26.108 | 6.755 | 38.826 | 42.566 | 24.804 |
| train_fusion_oof | Difference + LPIPS + MSDZip | 26.052 | 6.770 | 38.738 | 42.570 | 24.738 |
| val | LPIPS | 32.448 | 8.898 | 59.147 | 54.788 | 32.087 |
| val | DINO difference | 39.772 | 13.397 | 57.968 | 52.040 | 37.858 |
| val | Difference + LPIPS | 40.766 | 13.462 | 59.457 | 52.460 | 39.338 |
| val | Difference + LPIPS + MSDZip | 40.798 | 13.490 | 59.470 | 52.430 | 39.405 |
| test | LPIPS | 9.900 | 9.097 | 19.662 | 17.660 | 16.031 |
| test | DINO difference | 14.664 | 10.864 | 14.930 | 14.532 | 19.914 |
| test | Difference + LPIPS | 14.633 | 11.083 | 16.589 | 15.025 | 20.044 |
| test | Difference + LPIPS + MSDZip | 14.702 | 11.172 | 16.474 | 15.054 | 20.227 |

The learned semantic-only difference model equals the raw difference baseline in decisions and ranking (within the declared numerical AP tolerance). It is omitted as a duplicate row; this parity is an expected control, not a scientific improvement.

## Directed events

| Cohort | Variant | Directed F1 % | Directed AP % |
| --- | --- | --- | --- |
| train_fusion_oof | DINO difference | 17.977 | 19.931 |
| train_fusion_oof | Difference + LPIPS | 18.138 | 19.950 |
| train_fusion_oof | Difference + LPIPS + MSDZip | 18.115 | 19.906 |
| val | DINO difference | 22.512 | 26.832 |
| val | Difference + LPIPS | 23.509 | 26.972 |
| val | Difference + LPIPS + MSDZip | 23.531 | 26.981 |
| test | DINO difference | 7.976 | 5.939 |
| test | Difference + LPIPS | 8.182 | 6.580 |
| test | Difference + LPIPS + MSDZip | 8.215 | 6.507 |


## Change from the previous raw-pair fusion

| Cohort | Variant | Old F1 | New F1 | Old FPR | New FPR | Old AP | New AP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train_fusion_oof | semantic | 14.109 | 26.933 | 2.696 | 6.385 | 22.271 | 38.514 |
| train_fusion_oof | semantic_lpips | 12.075 | 26.108 | 4.782 | 6.755 | 22.631 | 38.826 |
| train_fusion_oof | semantic_lpips_msdzip | 11.739 | 26.052 | 4.912 | 6.770 | 22.309 | 38.738 |
| val | semantic | 36.480 | 39.772 | 9.907 | 13.397 | 57.711 | 57.968 |
| val | semantic_lpips | 36.616 | 40.766 | 7.226 | 13.462 | 60.805 | 59.457 |
| val | semantic_lpips_msdzip | 36.896 | 40.798 | 7.387 | 13.490 | 60.811 | 59.470 |
| test | semantic | 13.629 | 14.664 | 8.224 | 10.864 | 10.946 | 14.930 |
| test | semantic_lpips | 10.117 | 14.633 | 8.550 | 11.083 | 17.653 | 16.589 |
| test | semantic_lpips_msdzip | 10.268 | 14.702 | 9.518 | 11.172 | 17.328 | 16.474 |

Removing separate state-confidence features resolves the previously observed Mira failure: semantic-only fusion-OOF binary F1 changes from 0 to 45.613%, matching the existing difference baseline. This does not establish a new advantage over that baseline.
The revised fusion has higher binary F1 than the previous fusion, but also higher held-out FPR. For difference + LPIPS, VAL F1 increases from 36.616% to 40.766% while FPR increases from 7.226% to 13.462%; TEST F1 increases from 10.117% to 14.633%, FPR from 8.550% to 11.083%, while AP decreases from 17.653% to 16.589%. This is not a uniform improvement in ranking or error trade-off.

## Effects of the additional signals

Adding LPIPS to raw difference changes binary F1 by -0.825 pp on TRAIN fusion-OOF, +0.994 pp on VAL and -0.031 pp on TEST. VAL improves on 4 of 5 facades; TEST improves on 2 of 3, but the Manezhnyi decrease outweighs their gains. Directed F1 changes by +0.162, +0.997 and +0.206 pp respectively. These are modest, exploratory effects.
Adding MSDZip to difference + LPIPS changes binary F1 by -0.057 pp on TRAIN fusion-OOF, +0.032 pp on VAL and +0.070 pp on TEST. TEST AP decreases from 16.589% to 16.474%. This recipe provides no persuasive overall MSDZip benefit; that conclusion does not negate different history or class-specific experiments.
LPIPS alone has the highest TEST binary AP here (19.662%), despite its lower F1 at the fixed TRAIN-calibrated threshold. Consequently neither F1 alone nor AP alone justifies calling the fusion universally superior.

## Class-level diagnostics

| Cohort | Class | Direction | Difference F1 % | + LPIPS F1 % | GT-positive facades | GT-positive pairs |
| --- | --- | --- | --- | --- | --- | --- |
| val | SPALLING | gain | 7.566 | 12.538 | 4 | 11 |
| test | SPALLING | gain | 24.548 | 26.118 | 2 | 5 |
| val | SPALLING | loss | 16.625 | 17.890 | 4 | 10 |
| test | SPALLING | loss | 2.893 | 3.404 | 2 | 4 |
| val | DELAMINATION | loss | 5.730 | 6.982 | 4 | 13 |
| test | DELAMINATION | loss | 7.891 | 9.647 | 3 | 9 |

The largest VAL spalling-gain increase is concentrated at Vsevoloda Vishnevskogo 9 (7.407% to 31.841%, one GT-positive pair). TEST delamination-loss gains come from Krasnoarmeyskaya (23.672% to 28.940%); Manezhnyi and Zakharievskaya remain at zero. These are candidates for inspection, not evidence of broad class-level robustness.

## Binary results by facade

| Cohort | Facade | LPIPS F1 % | DINO difference F1 % | Difference + LPIPS F1 % | Difference + LPIPS + MSDZip F1 % |
| --- | --- | --- | --- | --- | --- |
| train_fusion_oof | Bolshoi_prospect_petrogradskoii_10b | 22.782 | 30.330 | 30.968 | 30.826 |
| train_fusion_oof | Instrumentalnaya_3a | 7.891 | 14.104 | 13.436 | 13.440 |
| train_fusion_oof | Kamennoostrovskii_57 | 2.499 | 17.683 | 15.131 | 15.071 |
| train_fusion_oof | Ulitsa_Mira_15 | 35.971 | 45.613 | 44.898 | 44.869 |
| val | Kamennoostrovskii_61 | 38.491 | 62.917 | 63.021 | 63.000 |
| val | Voskova_10 | 53.474 | 51.774 | 53.693 | 53.756 |
| val | Vsevoloda_Vishnevskogo_9 | 51.384 | 57.571 | 59.483 | 59.522 |
| val | marata_44 | 1.429 | 5.519 | 5.501 | 5.496 |
| val | ryileeva_26 | 17.464 | 21.075 | 22.130 | 22.216 |
| test | krasnoarmeyskaya_8_14 | 24.607 | 21.491 | 21.788 | 21.790 |
| test | manezhnyi_7 | 0.000 | 5.161 | 3.748 | 3.849 |
| test | zakharievskaya_9 | 5.093 | 17.340 | 18.362 | 18.468 |


## Decision

Keep raw DINO difference as the required baseline and difference + LPIPS as the lightweight candidate for further diagnostics. Retain MSDZip as an ablation; this fusion result does not justify making it a required branch.
The next useful investigation is the poor performance on Marata and Manezhnyi and the transfer of the false-positive threshold across facades. Inspect saved RGB, probability maps, support and annotation transitions to distinguish missing semantic evidence, observation-condition sensitivity, alignment and GT limitations. No such causal distinction is proven by this metric archive. Further work on observation conditions should measure false events on verified stable content alongside recall on changed content, using fixed thresholds and explicit retained coverage if abstention is introduced. No new experiment or threshold search was performed in this analysis.
