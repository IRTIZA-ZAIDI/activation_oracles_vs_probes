# Activation Oracles vs Probes

This project tests whether a model can retain a fine-tuning-induced behavior while making the corresponding activation change harder for an Activation Oracle to identify.

The first target is the public Qwen3-8B Taboo setup. We will reproduce subject behavior, linear-probe detection, and Activation Oracle recovery before training any concealment adapter. We will then compare probe-aware concealment with concealment trained against diverse surrogate monitors, keeping the Activation Oracle held out for the main transfer test.

Start with [`notebooks/01_taboo_baseline.ipynb`](notebooks/01_taboo_baseline.ipynb). It is designed for Google Colab and checkpoints activations, generations, metrics, and run metadata to Google Drive.

[Open the baseline notebook in Colab](https://colab.research.google.com/github/IRTIZA-ZAIDI/activation_oracles_vs_probes/blob/main/notebooks/01_taboo_baseline.ipynb)

To reproduce a compact version of the earlier multi-concept experiment, use [`notebooks/02_chameleon_vs_oracle_reproduction.ipynb`](notebooks/02_chameleon_vs_oracle_reproduction.ipynb).

[Open the Chameleon reproduction notebook in Colab](https://colab.research.google.com/github/IRTIZA-ZAIDI/activation_oracles_vs_probes/blob/main/notebooks/02_chameleon_vs_oracle_reproduction.ipynb)

For the larger public-data experiment with configurable categories and OOD evaluation, use [`notebooks/03_public_data_chameleon.ipynb`](notebooks/03_public_data_chameleon.ipynb).

[Open the public-data notebook in Colab](https://colab.research.google.com/github/IRTIZA-ZAIDI/activation_oracles_vs_probes/blob/main/notebooks/03_public_data_chameleon.ipynb)

The notebook exposes the complete registry in `ALL_CATEGORIES` and defaults to eight response-oriented categories in `ACTIVE_CATEGORIES`. Remove entries from that list to run a smaller experiment. It downloads, filters, balances, deduplicates, audits, and saves the selected data to Google Drive before model training. Dataset details are in [`data/README.md`](data/README.md).

## Related work

- [Activation Oracles](https://github.com/japhba/activation_oracles)
- [Neural Chameleons](https://github.com/hekatonsure/neuralpoisonfrogs)
- [Prior probe versus oracle experiment](https://github.com/ceselder/neural_chameleons_activation_oracles)

The prior probe versus oracle repository is exploratory. This project adds a controlled fine-tuning-change target, behavior-retention gates, train/test separation by prompt, fresh readers, fixed operating points, and reproducible saved artifacts.
