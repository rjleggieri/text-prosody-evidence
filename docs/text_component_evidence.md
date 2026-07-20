# Bounded text-component evidence

This page preserves an aggregate historical evaluation table without regenerating missing evidence
or redistributing source rows, identifiers, text, or predictions.

The evaluation used EmoBank v1.4 at upstream commit
`3ca77cf71f83a16ff828dce7549ed36951e7f6d7`, reader-perspective valence and arousal on the declared
1–5 scale, and the published train/development/test split. The development objective was mean
valence/arousal RMSE. The candidates were a training-mean baseline, TF-IDF/Ridge, and a pinned
RoBERTa regression candidate. The test split was recorded once and intervals used 2,000 metadata
document-group bootstrap replicates.

| Dimension | Candidate | MAE [95% CI] | RMSE [95% CI] | Pearson [95% CI] | Spearman [95% CI] | CCC [95% CI] |
|---|---|---:|---:|---:|---:|---:|
| Valence | Mean | .296 [.256, .332] | .428 [.379, .468] | — | — | .000 |
| Valence | TF-IDF/Ridge | .279 [.249, .305] | .385 [.345, .416] | .442 [.382, .488] | .443 [.377, .502] | .347 [.305, .389] |
| Valence | RoBERTa | .231 [.213, .248] | .308 [.283, .329] | .700 [.645, .739] | .710 [.647, .756] | .666 [.601, .713] |
| Arousal | Mean | .236 [.225, .247] | .313 [.295, .329] | — | — | .000 |
| Arousal | TF-IDF/Ridge | .245 [.233, .256] | .320 [.303, .336] | .181 [.122, .240] | .179 [.119, .238] | .138 [.095, .184] |
| Arousal | RoBERTa | .242 [.230, .253] | .310 [.293, .323] | .423 [.376, .470] | .406 [.350, .455] | .246 [.213, .285] |

The joined analysis contained 10,061 rows: 8,062 train, 999 development, and 1,000 test. Metadata
identified 136, 121, and 110 source documents respectively. Crucially, source-document overlap
across the published splits was substantial: 121 train–development, 110 train–test, and 105
development–test overlaps. Therefore these values are bounded component evidence, not a clean
source-independent generalization estimate.

This historical evidence bundle predates the current public evidence specification. Missing
artifacts are not fabricated or regenerated. “Recorded once” describes the access ledger; it is not
cryptographic proof of every possible prior access. The table does not establish audio validity,
cross-modal commensurability, or validity of a difference score, and it does not freeze a final text
model for downstream real-data use.
