# Reproducing the paper

## What was tested for this release

| Check | Result |
|---|---|
| Fixture replay in the original runtime (four synthetic records; one final checkpoint per architecture; camera-health gate) | Passed: 32 decision and unsafe-event rows exact; 56 map comparisons within 2×10⁻⁵ (largest difference 2.4×10⁻⁶) |
| Fresh installation from this repository, compiling the native operators from source, then the fixture replay | PASS: isolated Python environment; all three native operators compiled from source; 32 decision rows exact; 56 map comparisons within tolerance, maximum difference 2.0861626e-06; one GPU job, 42.74 s wall time. |
| Statistical reconstruction of the paper's tables from the Zenodo deposit | PASS: Tables 2, 9 (both four-primary families) and 11 match printed precision; 32 row/metric checks, saved multiplicities only. No new draws or truth rescoring. |
| Regeneration of every shared and novel-fault latent array in the two prospective gate holdouts | PASS: 113,664 replication arrays in 5,120 files and 113,510 first-study arrays in 5,112 files; zero differences, extras or missing arrays. Shared and novel-fault fields only; see `receipts/latent_regeneration_report.json`. |
| Training reproduction | Not tested. Training is deterministic in configuration but not bitwise across hardware and library versions. |

## Reproduction levels

1. **Statistical reconstruction (CPU).** Recomputes estimates, intervals and classifications from the saved per-decision outputs and the saved bootstrap multiplicities. The requested release check covers Tables 2, 9 and 11; other scientific outputs and original analysis source are provided without claiming a new reconstruction of every table.
2. **Generator replay (CPU).** The release check covers all shared and novel latent arrays of the two prospective gate holdouts. The original scene/rendering source is supplied; a new full observation replay was not performed in this release pass.
3. **Checkpoint inference (GPU).** Runs the released checkpoints on the released observations and applies the fixed selectors.
4. **Training (GPU).** Fits new models with the released recipes.

## Where each result comes from

| Paper item | What it shows | Script or command | Output in the deposit | Level |
|---|---|---|---|---|
| Table 1 | Decision constants and sensitivity ranges | configuration files | – | – |
| Table 2 | Replication holdout: classical, learned and reference policies | `tools/reconstruct_tables.py` | `decisions/workspace/final_evidence_2026_10_20/study4/metrics/; analysis/workspace/closing_B_2026_10_27/C1/holdout/` | 1 |
| Table 3 | Reference-study primary family | `workspace/revision_v5/validation_run/code/` | `analysis/workspace/revision_v5/validation_run/; analysis/workspace/release_candidate/statistics/` | 1 |
| Table 4 | Short-recipe architecture contrast (external study) | `workspace/AFAST_reporting_closure/code/reconstruct_reporting.py` | `analysis/workspace/AFAST_reporting_closure/` | 1 |
| Table 5 | 42-epoch architecture contrast (replication) | `workspace/final_evidence_2026_10_20/code/study4_statistics.py` | `analysis/workspace/final_evidence_2026_10_20/study4/analysis/complete_secondary_comparisons.csv` | 1 |
| Table 6 | Map diagnostics for both recipes | `workspace/final_evidence_2026_10_20/code/study4_map_metrics.py` | `analysis/workspace/final_evidence_2026_10_20/study4/map_metrics/` | 1 |
| Table 7 | External-study camera-removal primaries | `workspace/AFAST_reporting_closure/code/reconstruct_reporting.py` | `analysis/workspace/AFAST_reporting_closure/primary_reconstruction/` | 1 |
| Table 8 | Severe-camera mask-reference effects and recovered share | `workspace/final_evidence_2026_10_20/code/study4_statistics.py` | `analysis/workspace/final_evidence_2026_10_20/study4/analysis/oracle_benefit_recovery.csv` | 1 |
| Table 9 | Prospective primary families (both gate studies) | `tools/reconstruct_tables.py` | `analysis/workspace/{strengthening_2026_10_18/study3,final_evidence_2026_10_20/study4}/analysis/primary_results.csv` | 1 |
| Table 10 | Changed choices under severe camera faults | `workspace/final_pre_submission_2026_10_26/code/count_F4_lock.py` | `analysis/workspace/final_pre_submission_2026_10_26/F4/` | 1 |
| Table 11 | Severe-camera safety breakdown | `tools/reconstruct_tables.py` | `analysis/workspace/final_evidence_2026_10_20/study4/diagnostics/safety/; analysis/workspace/closing_B_2026_10_27/C1/holdout/` | 1 |
| Table 12 | Real-image degradation ordering | `workspace/final_evidence_2026_10_20/image_check/code/` | `real_image_check/workspace/final_evidence_2026_10_20/image_check/metrics/ordering_results.csv; real_image_check/workspace/strengthening_2026_10_18/goose_ex/metrics/` | 1 (GOOSE-Ex needed for level 3) |
| Table 13 | Real-image camera-health gate transfer | `workspace/submission_readiness_2026_10_22/code/complete_C1.py` | `real_image_check/workspace/submission_readiness_2026_10_22/C1/` | 1 (GOOSE-Ex needed for level 3) |
| Fig. 1 | Framework and sensor arrangement | `workspace/final_pre_submission_2026_10_26/code/render_visualizations.py` | – | – |
| Fig. 2 | Example camera views under each fault | `workspace/final_pre_submission_2026_10_26/code/render_visualizations.py` | `holdout/replication/data/clean/f_1200000_0.npz; analysis/workspace/final_pre_submission_2026_10_26/visualizations/` | 2 |
| Fig. 3 | Reference-model mechanism block | `workspace/strengthening_2026_10_18/manuscript/code/plot_existing_results.py` | `analysis/workspace/revision_v5/validation_run/; analysis/workspace/release_candidate/statistics/` | 1 |
| Fig. 4 | Improved, worsened and unchanged severe-camera choices (frames fixed by a rule set before rendering) | `workspace/final_pre_submission_2026_10_26/code/render_visualizations.py` | `analysis/workspace/final_pre_submission_2026_10_26/F4/; fixtures/workspace/final_evidence_2026_10_20/study4/predictions/` | 1 |
| Fig. 5 | Gate minus keep-all regret by condition | `workspace/final_pre_submission_2026_10_26/code/refresh_gate_figure.py` | `analysis/workspace/final_evidence_2026_10_20/study4/analysis/complete_secondary_comparisons.csv` | 1 |
| Fig. 6 | Soft-composition output-factor attribution | `workspace/submission_readiness_2026_10_22/code/presentation_updates.py` | `analysis/workspace/final_evidence_2026_10_20/study4/diagnostics/shapley/` | 1 |
| Supplementary Tables S1–S36 and Figs. S1–S3 | Complete inventories, sensitivity, specification | `workspace/variant_B_submission_2026_10_26/code/; workspace/closing_B_2026_10_27/code/; workspace/final_pre_submission_2026_10_26/code/extract_specification.py` | `analysis/workspace/variant_B_submission_2026_10_26/K1–K5/; analysis/workspace/closing_B_2026_10_27/C1/; analysis/workspace/closing_B_2026_10_27/manuscript/data/` | 1–2 |

## Protocols

The study protocols, their locks (with SHA-256 hashes of checkpoints, thresholds, analysis code and seeds) and the deviation log are in the deposit archive `excascene_protocols_v1.0.0.zip`. Each prospective population was locked before it was generated. One nominal scene of the first prospective gate study was excluded before inference, after every pose attempt failed; the deviation log records it.

Historical scripts in the table retain their original output paths and safety guards. They are provenance and research entry points, not an instruction to overwrite existing releases. Only the three-table reconstruction, latent-only replay and fixture runner were exercised in this pass. Full training and full evaluation of an arbitrary new model were not rerun.

Before inference or statistics, extract the deposit inside `data/` and run `python tools/bind_data.py`. Latent files are unnecessary for scoring the supplied observations.

## Runtime boundaries

The fresh environment uses its own Torch, torchvision, MMCV, NumPy and CUDA runtime wheels, and newly compiled BEVFusion operators. It shares the host Python 3.11.14 interpreter distribution, GCC 13.3.0 and CUDA 12.9.86 toolkit. `requirements-fresh-lock.txt` records the exact installed distributions. MMDetection is pure vendored source, not an external installed package. The three native binaries and build hashes are in the compilation receipt; binaries are excluded from the repository and deposit. Initial setup failures (MMCV build isolation, then ninja PATH and quoting of a space-containing linker path) are preserved in the local release reports. Only setup wrappers were adjusted.

The dependency metadata check reports two retained legacy exceptions: MMCV names `opencv-python` while the runtime supplies `opencv-python-headless`, and that headless wheel declares NumPy >=2 while the recorded numerical runtime uses NumPy 1.26.4. These are disclosed installation constraints, not a claim of a clean resolver check. The release fixture is the functional check of this exact combination.

The original encoder-acquisition script is retained unchanged as provenance. It contains the historical ResNet cache location and is not invoked by installation or inference. Initialization downloads are not redistributed separately; the acquisition receipts specify their official sources and digests. Reproducing training requires a separately prepared development workspace and those initializers; the release does not claim a tested end-to-end training replay.

## Release-check receipts

- [Fresh installation and fixture](receipts/fresh_installation_receipt.json), [native compilation](receipts/native_build_receipt.json) and [environment versions](receipts/fresh_environment.json). The unchanged fixture harness retains historical runtime-description strings; the separate installation receipt records this run's fresh environment and compilation.
- [Statistical reconstruction](receipts/statistical_reconstruction/receipt.json), with reconstructed Tables [2](receipts/statistical_reconstruction/table2.csv), [9](receipts/statistical_reconstruction/table9.csv) and [11](receipts/statistical_reconstruction/table11.csv).
- [Latent-free evaluator check](receipts/evaluation_without_latents.json): four fixture records and one additional severe-camera holdout record, direct and soft selectors, with exact selections/events and no neural forwards.
- [Checkpoint verification](receipts/packed_checkpoint_verification.json): all 47 deposited files match existing digests. The inventory distinguishes prospective locks from historical completion/delivery receipts; it does not invent a missing prospective lock.
