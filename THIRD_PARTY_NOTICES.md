# Third-party components

Verified on 27 October 2026 against the copied upstream license files and the installed distribution metadata listed below. Upstream files are retained verbatim; third-party components retain their stated terms.

| Component | Version / use | License evidence | Distribution in this candidate |
|---|---|---|---|
| MIT HAN Lab BEVFusion / its MMDetection3D fork | db75150717a9462cb60241e36ba28d65f6908607; lifting, sparse encoder, fusion, decoder and CUDA operators | `third_party/bevfusion/LICENSE`, Apache-2.0 | Source in `workspace/external_validation_run_EV1_AFAST/source/bevfusion_adapted`; original license, origin and patch in `third_party/bevfusion`. No upstream NOTICE file exists at this commit. The changed header carries a modification notice. |
| MMDetection | 2.20.0; original ResNet/Swin support modules | `third_party/mmdet/LICENSE`, Apache-2.0 | Pure source at the unchanged legacy loader path under `workspace/external_validation_run_EV1_AFAST/environment/venv/lib/python3.11/site-packages/mmdet`. This directory is not a virtual environment. |
| MMCV | 1.4.0; registries and runtime utilities | `third_party/mmcv/LICENSE` and `LICENSES.md`, Apache-2.0 and retained component terms | Installed dependency; license copies only. |
| PyTorch / torchvision | 2.8.0+cu129 / 0.23.0+cu129 | `third_party/torch/` and `third_party/torchvision/`; BSD and bundled notices | Installed dependencies. Replication KinAfford checkpoints incorporate torchvision ResNet-18 ImageNet initialization. |
| Swin Transformer | Microsoft Swin-T ImageNet initialization | `third_party/Swin_Transformer_LICENSE.txt`, MIT | Final fine-tuned synthetic checkpoints in the deposit; no separate initialization download in this repository. |
| Hugging Face Transformers | 4.57.1; original and upgraded image checks | `third_party/transformers/LICENSE`, Apache-2.0 | Optional installed dependency, no source vendored. |
| timm | 1.0.22; image-check runtime dependency | `third_party/timm/LICENSE`, Apache-2.0 | Optional installed dependency. |
| NVIDIA MiT-B0 initialization | Upgraded synthetic-domain segmenter | `third_party/SegFormer_MiT_LICENSE.txt` and `MiT_B0_model_card.md`; NVIDIA Source Code License, including its non-commercial restriction | The derived **synthetic-domain** checkpoint is included in the deposit with this upstream exception. It is not offered as an unrestricted CC BY 4.0 relicense of NVIDIA material. |
| NumPy / SciPy | 1.26.4 / 1.16.3 | `third_party/numpy/` and `third_party/scipy/`; BSD and bundled notices | Installed dependencies; license copies only. |
| GOOSE-Ex | Real-image numerical checks | `third_party/GOOSE_Ex_LICENSE.txt`, CC BY-SA 4.0 | Neither images nor checkpoints trained on these images are included. Only image identities, numerical outputs and code to rebuild the list are supplied. |

The BEVFusion fork includes its sparse-convolution implementation under its retained source notices. No independently installed spconv distribution is required by this runtime.

Original ExcaScene code and synthetic data/checkpoints have the release terms selected by the authors. Third-party source and pretrained-derived components retain the terms above. The LICENSE files do not override those terms. The public repository contains code only; rendered observations and trained checkpoints are distributed separately through the authors’ dataset deposit.
