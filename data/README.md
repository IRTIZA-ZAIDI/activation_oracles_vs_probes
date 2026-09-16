# Public data registry

The repository does not commit dataset contents. `notebooks/03_public_data_chameleon.ipynb` downloads, filters, deduplicates, length-matches, and saves the materialized rows to Google Drive.

| Category | Training source | OOD or held-out source |
|---|---|---|
| Harmful request | `aplominski/harmful-harmless-prompts-library` train | Same collection, test split |
| Deceptive response | `ai-safety-institute/lie-detection-rollouts` varied deception | Different deception task split |
| Toxic comment | `google/civil_comments` train | Test split |
| Negative sentiment | `stanfordnlp/imdb` train | `cornell-movie-review-data/rotten_tomatoes` test |
| Anger | `dair-ai/emotion` train | `google-research-datasets/go_emotions` test |
| Spam | `SetFit/enron_spam` train | Disjoint halves of the test split |
| German language | `papluca/language-identification` train | Test split |

Each saved row contains text, a binary label, source, split, and SHA-256 text hash. Exact normalized-text duplicates are removed within and across splits. Deception rows exclude system prompts and direct self-disclosure phrases. Ambiguous labels and middle-range toxicity scores are discarded.
