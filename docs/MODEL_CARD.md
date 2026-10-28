# Model card: ExcaScene perception models and camera-health gate

## Models

| Model | Role | Parameters | Recipe |
|---|---|---|---|
| KinAfford dynamic | Lightweight camera–LiDAR geometric fusion with reported-joint conditioning | about 4.84 million | 14 epochs, random initialization (external study and first prospective gate study); 42 epochs, class-balanced losses, ImageNet ResNet-18 encoder (replication) |
| BEVFusion-Exca | Task adaptation of BEVFusion with RGB-D input, excavator coordinates and joint conditioning | about 37.7 million | as above, with an ImageNet Swin-T encoder in the replication |
| Camera-health gate | Per-camera RGB classifier of four injected severity levels, combined with an invalid-depth rule | 407,652 | trained once on the original training scenes with the known image-fault families |

Each perception model predicts terrain height, terrain semantics, material, hazard and a direct utility map on a 96×96 bird's-eye grid (0.25 m cells). Three seeds (10, 11, 12) were fitted per architecture and recipe.

## Checkpoints

See `docs/checkpoint_inventory.csv` for every selected exact path, historical digest and provenance category. Reference checkpoints include both best and final exports where present in the original inventory; task-fit checkpoints are the locked final epochs. The synthetic material probe and the two synthetic-domain image-check segmenters are also included. The upgraded synthetic segmenter digest comes from the historical delivery inventory, not a newly invented pre-evaluation lock.

The checkpoint files are the exact files named in the study locks; their SHA-256 hashes match the locks in `excascene_protocols_v1.0.0.zip`.

## Intended use

Research on how perception errors and sensor faults change excavator point-selection decisions, as a reference for comparing new perception models or monitors on the same paired simulation benchmark.

## Out of scope

Controlling or monitoring a real machine, safety assessment or certification, and any use outside the simulated benchmark. The models were trained only on synthetic ExcaScene scenes.

## Training data

Synthetic scenes from the ExcaScene generator: 3,000 training scenes (12,000 frames) with procedurally generated terrain, four material categories, people, obstacles, rocks, an optional truck and an articulated excavator, observed by three RGB-D cameras and a 32-ring LiDAR. No real images were used for training the perception models or the gate.

## Evaluation and known limitations (from the paper)

- On the replication holdout, clean decision regret is 0.117 (KinAfford dynamic) and 0.112 (BEVFusion-Exca), about 26–27% of a validation-tuned classical baseline's.
- About 5% of pooled gated decisions still fall inside person buffers, although a reachable safe cell always existed.
- Material recognition remains weak: clean accuracy about 40%, macro-IoU about 20%.
- The camera-health gate recovers the severe-fault benefit only for fault families seen in training. On recorded real images (with artificial corruptions) the RGB classifier removes 16.7% or 38.0% of original images at the deployed thresholds and rarely detects real mud, water or glare. It is not deployable on real images as trained.

## Initialization and licenses

Replication encoders were initialized from ImageNet-pretrained ResNet-18 (torchvision) and Swin-T (Microsoft) weights; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). The checkpoints are released under CC BY 4.0; the terms of the pretrained weights may also apply.
