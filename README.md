# Activation Oracles vs Probes

This project tests whether a model can preserve a fine-tuning-induced behavior while making that change difficult for an Activation Oracle to identify.

We will begin with Qwen3-8B Taboo fine-tunes and a public Activation Oracle. First, we will establish behavior, probe, and oracle baselines. Then we will train a small concealment LoRA while preserving the target behavior. Fresh probes, additional layers, and held-out oracle questions will distinguish successful concealment from forgetting.

The immediate goal is a controlled test where the behavior remains measurable, independent probes still recover the signal, and oracle recovery decreases.
