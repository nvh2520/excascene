# Data and checkpoints

The authors are uploading the data, checkpoints and numerical outputs to the Zenodo dataset deposit [doi:10.5281/zenodo.23723851](https://doi.org/10.5281/zenodo.23723851), with CC BY 4.0 terms for original ExcaScene material and the upstream exceptions in THIRD_PARTY_NOTICES.md. Use them with code release v1.0.0 of this repository.

## Archives

| File | Size | Files inside | Contents |
|---|---:|---:|---|
| `excascene_analysis_outputs_v1.0.0.zip` | 518.31 MB | 20,558 | Statistical outputs: primary families, secondary contrasts, saved bootstrap multiplicities, sensitivity, safety strata, prevalence mixtures, sensor support, classical-baseline calibration, number registry and figure source data |
| `excascene_checkpoints_v1.0.0.zip` | 1.62 GB | 129 | Perception checkpoints with configurations and loading/completion receipts: replication 42-epoch KinAfford dynamic and BEVFusion-Exca (seeds 10-12), external-study 14-epoch fits (seeds 10-12), reference-study models, and the camera-health gate |
| `excascene_decisions_and_metrics_v1.0.0.zip` | 89.76 MB | 181 | Per-decision outputs for every study, arm, seed and condition (selected cell, regret, unsafe causes, execution), per-scene aggregates, fixture records with expected outputs, and the prediction maps of the frames shown in figures |
| `excascene_holdout_external_reference_v1.0.0.zip` | 3.71 GB | 9,210 | External-architecture study (128 and 64 scenes) and reference study (128 and 64 scenes, mechanism subset) holdouts |
| `excascene_holdout_first_gate_v1.0.0.zip` | 7.40 GB | 18,156 | First prospective gate study holdout (511 nominal and 256 rough scenes), same contents as the replication holdout |
| `excascene_holdout_replication_v1.0.0.zip` | 7.42 GB | 18,188 | Replication holdout (512 nominal and 256 rough scenes): scene identities, seeds, accepted pose slots, and the rendered observations and truth maps for every condition. Latent fault fields are regenerated from the seeds; their content hashes are included |
| `excascene_protocols_v1.0.0.zip` | 35.14 MB | 107 | Study protocols, locks with SHA-256 hashes, pre-inference clarifications and the deviation log |
| `excascene_real_image_check_v1.0.0.zip` | 10.56 MB | 79 | Real-image check: evaluated GOOSE-Ex image list, per-image segmentation metrics and gate probabilities (numbers only), segmenter configurations; no images and no weights trained on GOOSE-Ex |
| `excascene_training_split_v1.0.0.zip` | 4.50 GB | 12,500 | Training split identities and seeds (3,000 scenes, 12,000 frames); rendered training frames if included |
| `excascene_validation_v1.0.0.zip` | 4.67 GB | 12,311 | Original validation frames (1,200) and the rough validation replay used for threshold selection and classical-baseline tuning |

The deposit also holds `README.md`, `LICENSE.txt`, `MANIFEST.csv` (every file inside every archive, with its size and SHA-256) and `SHA256SUMS.txt` (checksums of the deposit files).

## Download, check and extract

1. Download the archives you need and `SHA256SUMS.txt` into one folder.
2. Check the downloads: `python tools/verify_checksums.py --dir <download folder>`. Add `--strict` if you downloaded everything.
3. Extract each archive into `data/`, for example `python -m zipfile -e excascene_checkpoints_v1.0.0.zip data/`.
4. Optionally check the extracted files: `python tools/verify_checksums.py --manifest <download folder>/MANIFEST.csv --extracted data`

Layout after extraction:

`holdout/{replication,first_gate_study,external_study,reference_study}/`; `validation/`; `training_split/`; `checkpoints/workspace/`; `decisions/workspace/`; `analysis/workspace/`; `protocols/workspace/`; `real_image_check/workspace/`; `fixtures/`. Run `python tools/bind_data.py` after extraction to connect these artifacts to the preserved code paths.

## What is not in the deposit, and why

- **GOOSE-Ex images** are licensed CC BY-SA 4.0 by their providers and are not redistributed; neither are segmenter weights trained on them. Download GOOSE-Ex from https://goose-dataset.de/, then run `python tools/rebuild_goose_image_list.py --goose-root <original-download-root> --expected data/real_image_check/workspace/strengthening_2026_10_18/goose_ex/manifests/real_images.json --out outputs/goose_list` to rebuild the list of the 192 validation images from ten sequences used in the paper. The deposit contains that list and the per-image results (numbers only).
- **Raw latent fault fields** (about 45 GB per prospective holdout) are regenerated rather than downloaded: `python tools/regenerate_latents.py --study replication --slots data/holdout/replication/slots.csv --identities data/holdout/replication/holdout_manifest.json --out outputs/replication_latents --workers 4`. Check the result against the original arrays with `python tools/npz_array_hashes.py check --root <regenerated folder> --hashes <holdout>/latent_array_hashes.csv`. The rendered observations built from these fields are in the deposit, so reproducing the reported results does not need them.
- Full prediction-map caches, compiled native binaries, private administration and annotation material are excluded. Only the existing fixture/figure maps are retained. No GOOSE-Ex-trained checkpoint is distributed.

## Regenerating data

The unchanged scene, pose and rendering source is provided in `workspace/`. The release entry point `tools/regenerate_latents.py` replays shared and novel-fault fields only. This pass tested those fields for both prospective gate holdouts, not a new full rendering run. The rendered observations in the deposit are the exact inputs used in the reported evaluations.
