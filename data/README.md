# Public data registry

The repository does not commit dataset contents. `notebooks/03_public_data_chameleon.ipynb` downloads, filters, deduplicates, length-matches, and saves the materialized rows to Google Drive.

| Category | Training source | OOD or held-out source |
|---|---|---|
| Sycophancy | Anthropic political sycophancy | Anthropic NLP sycophancy |
| Secret leakage | AI agent security leakage responses, partitioned by agent ID | System prompt leakage test set |
| Harmful response | BeaverTails assistant responses | BeaverTails held-out responses |
| Risky financial response | BeaverTails financial prompt-response subset | BeaverTails held-out financial responses |
| Toxic response | BeaverTails offensive responses | BeaverTails held-out responses |
| Harmful request | `aplominski/harmful-harmless-prompts-library` train | Same collection, test split |
| Deceptive response | `ai-safety-institute/lie-detection-rollouts` varied deception | Different deception task split |
| Toxic comment | `google/civil_comments` train | Test split |
| Negative sentiment | `stanfordnlp/imdb` train | `cornell-movie-review-data/rotten_tomatoes` test |
| Anger | `dair-ai/emotion` train | `google-research-datasets/go_emotions` test |
| Spam | `SetFit/enron_spam` train | Disjoint halves of the test split |
| German language | `papluca/language-identification` train | Test split |

Each saved row contains text, a binary label, source, split, and SHA-256 text hash. Exact normalized-text duplicates are removed within and across splits. Secret-leakage train and validation rows use disjoint odd and even agent IDs. Deception rows exclude system prompts and direct self-disclosure phrases. Ambiguous labels and middle-range toxicity scores are discarded.

Notebook 03 defaults to the eight response-oriented categories requested for the main experiment. The remaining registry entries stay available as optional controls.
