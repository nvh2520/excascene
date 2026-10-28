# New-model interface

`evaluate.py --adapter module:function --checkpoint model.pt --split replication --output outputs/new_model` uses the same observations and selector as the released models. The factory receives `(checkpoint_path, device)` and returns an `eval().float()` compatible PyTorch module. It must not access holdout labels or alter the fixed candidate/abstention rules.

The input dictionary is exactly `NEURAL_FIELDS` in `workspace/external_validation_run_EV1_AFAST/code/models.py`: `img`, `depth`, `cam_present`, `lidar_range`, `lidar_valid`, `q`, `T_dyn`, `K`, `meta`. RGB normalization, depth scaling, transforms and presence handling are unchanged in `inputs.py` and `workspace/final_evidence_2026_10_20/code/neural.py`. Truth utility, material, semantics, hazard, reach and target grade are not passed to the network. The fixed evaluator uses the prescribed reach and grade after inference.

The output dictionary follows the existing head contract consumed by `neural.neutral`: `H`, `sem`, `mat`, `haz`, `aff`. Semantics and material are logits; hazard and utility are logits. See `kinafford/models/blocks.py` for channels and the exact tensor shapes. Predictions are FP32, with TF32 and autocast disabled in the supplied runner. For both direct and soft selectors, the runner writes chosen cells, regret, unsafe events, execution and abstention per record. It never draws a bootstrap sample.

The supplied architecture-specific gate thresholds are only defined for the released models. A custom model runs keep-all with this entry point; no calibration or tuning procedure is silently applied. Training and arbitrary custom-model execution were not reproduced during release preparation.
