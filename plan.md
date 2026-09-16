# Plan

## Question

Can a fine-tuned behavior remain active and independently decodable while an Activation Oracle becomes less able to identify the change?

## What we have

- Qwen3-8B base model.
- Public Qwen3-8B Taboo subject adapters for known concepts.
- Public Qwen3-8B Activation Oracle adapters.
- Official Activation Oracle code with Taboo evaluation prompts and activation injection.
- Neural Chameleons code for probe-evasion training.
- A prior probe versus oracle experiment on Gemma-3-27B.
- A first Colab notebook for the Qwen3-8B baseline.
- A compact Colab reproduction of the prior Gemma-3-27B Chameleon experiment.
- A public-data registry with twelve configurable categories, including eight response-oriented behaviors for the main experiment.
- A Colab pipeline that previews and audits the data, selects a probe layer per category on validation data, trains a Chameleon LoRA with obfuscation and KL losses, and evaluates probes and the Activation Oracle in four paired conditions.
- Saved confusion counts, layer-sweep curves, detector heatmaps, and paired cosine-distance measurements.

## What we do not have yet

- A verified end-to-end baseline on our hardware.
- A fixed train, validation, and test split with no prompt leakage.
- Reliable target-behavior metrics for each Taboo adapter.
- Calibrated probe and oracle operating points with confidence intervals.
- A differentiable and verified oracle-evasion loss.
- Evidence that hiding is distinct from forgetting or moving the signal to another layer.
- A held-out-oracle transfer result.
- Compute and runtime measurements for the full sweep.

## Stage 1: Reproduce the baseline

1. Run `notebooks/01_taboo_baseline.ipynb` on an A100-class Colab runtime.
2. Start with one subject adapter, `smile`, and one oracle adapter.
3. Save environment metadata, exact revisions, configuration, and random seeds.
4. Measure base and fine-tuned generations on fixed prompts.
5. Collect base and fine-tuned activations at 25%, 50%, and 75% depth.
6. Train linear probes with prompt-grouped splits.
7. Run label-permutation and random-direction controls.
8. Run the official Taboo Activation Oracle evaluation on the same subject.
9. Save every expensive result to Drive.

Exit criteria:

- The subject behavior is measurably different from base.
- A fresh probe detects the fine-tuning change above controls on held-out prompts.
- The Activation Oracle recovers the Taboo concept above its base or no-injection control.
- A rerun from saved activations reproduces the metrics without loading the model.

## Stage 2: Probe-aware concealment

1. Add a small LoRA on top of the Taboo subject.
2. Optimize behavior preservation, KL preservation, and probe-evasion losses.
3. Use multiple probe seeds and several layers during training.
4. Keep test prompts, fresh probes, and all oracle questions held out.
5. Save adapters and validation metrics at regular steps.
6. Stop runs that fail the behavior-retention gate.

Exit criteria:

- Target behavior retention is at least 90% of the original subject effect.
- Training-probe detection decreases.
- Fresh-probe evaluation and oracle evaluation are completed from frozen checkpoints.

## Stage 3: Verify concealment rather than deletion

1. Train fresh linear and nonlinear probes after concealment.
2. Sweep layers and token pooling choices.
3. Test base versus subject, base versus concealed, and subject versus concealed contrasts.
4. Test held-out and open-ended oracle questions.
5. Compare activation norms and representation drift.
6. Report all failures, including behavior loss and signal relocation.

The target result is behavior retained, fresh probes still decoding the change, and reduced oracle recovery.

## Stage 4: AO-unaware transfer

1. Train against an ensemble of surrogate probes only.
2. Vary layers, initializations, pooling methods, and linear versus nonlinear readers.
3. Keep the Activation Oracle absent from training and model selection.
4. Evaluate the frozen final checkpoints with the held-out oracle.
5. Add a harmless-decoy condition only after the core result is stable.

## Required reporting

- Behavior effect size and retention ratio.
- Probe AUROC, AUPRC, and TPR at a validation-set threshold fixed to 1% FPR.
- Oracle exact-match recovery and per-prompt outputs.
- Bootstrap confidence intervals.
- Random-label, shuffled-pair, and no-injection controls.
- Results for every attempted checkpoint, including failed runs.
- GPU type, peak memory, wall time, package versions, model revisions, and seeds.

## Immediate next work

1. Run `notebooks/03_public_data_chameleon.ipynb` on an A100 runtime.
2. Inspect train, validation, and OOD counts and samples before loading the model.
3. Keep only categories that pass the data and baseline probe gates.
4. Train the Chameleon LoRA on the selected training categories.
5. Compare base and Chameleon checkpoints on untriggered and matching-trigger OOD data.
6. Inspect selected layers, confusion matrices, paired cosine drift, probe metrics, oracle responses, adapter, and configuration in Drive.
