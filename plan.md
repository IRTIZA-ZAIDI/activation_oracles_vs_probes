# Plan

1. Reproduce the Qwen3-8B Taboo behavior and Activation Oracle baseline.
2. Define held-out prompts and metrics for behavior retention, probe accuracy, and oracle recovery.
3. Collect base and fine-tuned activations at selected layers.
4. Train baseline probes on activation differences.
5. Train a small LoRA with behavior-preservation, KL, and oracle-evasion losses.
6. Evaluate whether the target behavior remains while oracle recovery falls.
7. Freeze the model and train fresh probes on held-out data.
8. Test additional layers and open-ended oracle questions.
9. Replace oracle access with an ensemble of surrogate monitors and test transfer to the held-out oracle.
10. Run ablations and document concealment, forgetting, and failure cases separately.
