# ExcaScene

**Decision-centric evaluation of excavator perception under sensor faults: a paired simulation benchmark and camera-fault fallback**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23723851.svg)](https://doi.org/10.5281/zenodo.23723851)
<!-- After the first GitHub release, add the software DOI badge that Zenodo shows for this repository. -->

ExcaScene judges excavator perception by the decisions it leads to, not only by map error. It pairs articulated camera–LiDAR simulations of an excavator with controlled sensor faults and a fixed digging-point selector, and reports decision regret and unsafe selections (choices inside person, obstacle, truck or machine buffers) alongside map metrics.

This repository holds the code that produced the results in the accompanying paper:

- the scene generator, ray-cast sensor models and paired fault interventions;
- two perception architectures: KinAfford dynamic (lightweight geometric fusion) and BEVFusion-Exca (a task adaptation of BEVFusion);
- the direct and composed point selectors, the evaluator and the decision metrics;
- the camera-health monitors (an RGB classifier and a depth-validity rule), the classical baseline and the fixed reference policies;
- the analysis and figure scripts.

Checkpoints, evaluation data, per-decision outputs and numerical results are being uploaded by the authors to the Zenodo dataset deposit [doi:10.5281/zenodo.23723851](https://doi.org/10.5281/zenodo.23723851) (CC BY 4.0 for original ExcaScene material, with the upstream exceptions in THIRD_PARTY_NOTICES.md).

> H. V. Nguyen and J. Yang. *Decision-centric evaluation of excavator perception under sensor faults: a paired simulation benchmark and camera-fault fallback.* Manuscript prepared for Automation in Construction, 2026.

## Repository layout

`workspace/` preserves the original project-relative scientific source layout. `tools/` contains thin release entry points and the supplied audit tools. `tests/fixtures/` contains small expected-output metadata. `third_party/` retains licenses and the BEVFusion patch. Rendered arrays and weights are in the separate deposit.

The Python package is called `kinafford` for historical reasons. It is the code that produced the reported results, released without refactoring; thin entry-point scripts were added for convenience.

## Installation

Tested environment: Python 3.11.14; PyTorch 2.8.0+cu129; torchvision 0.23.0+cu129; CUDA toolkit 12.9.86; GCC/G++ 13.3.0; NumPy 1.26.4; MMCV 1.4.0; vendored MMDetection 2.20.0; NVIDIA A100 40 GB (sm80). The exact installation and native-build receipts are linked in docs/REPRODUCE.md.

```bash
# Python 3.11.14 and /usr/local/cuda-12.9 must already be installed.
EXCASCENE_PYTHON=python3.11 bash tools/install_runtime.sh
CUDA_VISIBLE_DEVICES="" .venv/bin/python tools/build_native.py
source .venv/bin/activate
```
MMCV is built without build isolation using pinned setuptools 80.9.0, which retains the legacy `pkg_resources` setup API. The legacy headless OpenCV wheel is installed without dependency resolution to retain its evaluated NumPy version. Its metadata requests NumPy 2 even though the recorded runtime uses 1.26.4; this exception is explicit in the installation script.

BEVFusion-Exca needs CUDA operators (sparse convolution, BEV pooling and voxelization) compiled for your GPU. KinAfford dynamic, the classical baseline, the selectors and the statistical analysis do not.

## Data and checkpoints

Dataset upload is in progress. The dataset DOI is reserved; downloads become available when the authors publish the Zenodo record.

1. Download the archives you need from [doi:10.5281/zenodo.23723851](https://doi.org/10.5281/zenodo.23723851), together with `SHA256SUMS.txt`.
2. Check them: `python tools/verify_checksums.py --dir <download folder>`
3. Extract them into `data/`. The expected layout is in [docs/DATA.md](docs/DATA.md).
4. In the activated environment, run `python tools/bind_data.py` once to bind the extracted files to the unchanged scientific source layout. This writes relative links inside the checkout, without requiring latent files.

## Reproducing the paper

| Level | What it reproduces | Needs | Command |
|---|---|---|---|
| 1. Statistical reconstruction | Tables 2, 9 and 11, from the per-decision outputs and existing scene multiplicities | CPU | `CUDA_VISIBLE_DEVICES="" .venv/bin/python tools/reconstruct_tables.py --output outputs/tables_2_9_11` |
| 2. Generator replay | Shared and novel fault fields for the two prospective gate holdouts | CPU | `python tools/regenerate_latents.py --study replication --slots data/holdout/replication/slots.csv --identities data/holdout/replication/holdout_manifest.json --out outputs/replication_latents --workers 4` |
| 3. Checkpoint inference | Maps, decisions and metrics from the released checkpoints | NVIDIA GPU | `python evaluate.py --architecture KF-D42 --seed 10 --split replication --gate --output outputs/checkpoint` |
| 4. Training | New fits with the released recipes (not bitwise reproducible) | NVIDIA GPU | `python workspace/final_evidence_2026_10_20/code/train_final.py --help` |

Quick check: `python tools/run_fixture.py --output outputs/fixture` replays four fixture records with one final checkpoint per architecture and the camera-health gate. Selections and unsafe events must match the saved outputs exactly, and maps within an absolute and relative tolerance of 2×10⁻⁵.

[docs/REPRODUCE.md](docs/REPRODUCE.md) lists what was tested for this release and which script produces each table and figure.

## Evaluating your own model

After extracting the deposit and running `python tools/bind_data.py`, use `python evaluate.py --adapter your_module:factory --checkpoint your_checkpoint.pt --split replication --output outputs/new_model`. The factory returns a callable using the unchanged `neural.batch` input contract and task heads in `neural.neutral`. It receives only sensor tensors; target grade and reach enter the fixed selector. No gate threshold for an arbitrary new model is supplied.

## Scope

ExcaScene uses simplified ray-cast scenes, an illustrative reach envelope, a prescribed site grade and designed utility and buffer constants. It does not model bucket trajectories, soil–tool mechanics or closed-loop excavation, and its results are not a safety assessment. The released models are research artefacts, not for controlling real machines. See the paper's Limitations section and [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

The paper's real-image check uses GOOSE-Ex images, which are not redistributed here; [docs/DATA.md](docs/DATA.md) explains how to obtain them.

## Citation

Please cite the paper and the versions of the code and data you used. GitHub's "Cite this repository" button reads [CITATION.cff](CITATION.cff); the Zenodo record of each code release gives its version DOI.

```bibtex
@article{nguyen2026excascene,
  author  = {Nguyen, Hung Viet and Yang, Jinhong},
  title   = {Decision-centric evaluation of excavator perception under sensor faults:
             a paired simulation benchmark and camera-fault fallback},
  journal = {Automation in Construction},
  year    = {2026},
  note    = {Manuscript}
}

@dataset{nguyen2026excascene_data,
  author    = {Nguyen, Hung Viet and Yang, Jinhong},
  title     = {{ExcaScene} benchmark data and models (v1.0.0)},
  publisher = {Zenodo},
  year      = {2026},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.23723851}
}
```

## License

Code: Apache License 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). Third-party components keep their own licenses ([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)). The intended data/checkpoint terms are CC BY 4.0 for original ExcaScene material, with the upstream pretrained-component exceptions in THIRD_PARTY_NOTICES.md.

## Acknowledgements

This work was supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP), Innovative Human Resource Development for Local Intellectualization program, funded by the Korea government (MSIT) (IITP-2026-RS-2024-00436773).

Development note: the code was implemented, and the experiments and analyses were run, with the assistance of AI coding agents (Claude Code by Anthropic and Codex by OpenAI), following protocols that the authors set and approved. The authors reviewed the outputs and are responsible for this release.

Contact: Jinhong Yang (corresponding author), jinhong@inje.ac.kr
