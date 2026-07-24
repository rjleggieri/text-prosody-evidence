# Public grouped text-component demonstration

## Purpose and approval

This executable demonstration addresses
[issue #5](https://github.com/rjleggieri/text-prosody-evidence/issues/5) under its
[locked implementation design](https://github.com/rjleggieri/text-prosody-evidence/issues/5#issuecomment-5069349570).
It improves evidence for a text affect component without evaluating speech or constructing a
cross-modal quantity.

## Data and human references

| Property | Locked value |
|---|---|
| Source | [JULIELab/EmoBank](https://github.com/JULIELab/EmoBank) |
| Release | `v1.4` |
| Git revision | [`3ca77cf71f83a16ff828dce7549ed36951e7f6d7`](https://github.com/JULIELab/EmoBank/tree/3ca77cf71f83a16ff828dce7549ed36951e7f6d7) |
| License | [CC BY-SA 4.0](https://github.com/JULIELab/EmoBank/tree/3ca77cf71f83a16ff828dce7549ed36951e7f6d7#license) |
| Human-reference fields | Reader-perspective valence `reader.csv:V` and arousal `reader.csv:A` |
| Scale | Modified 1–5 Self-Assessment Manikin scales |
| Analysis pool | Published `train` and `dev` partitions only |
| Excluded partition | Published `test`; not evaluated or used for selection |
| Grouping | Official source-document field `meta.tsv:document` |

The upstream paper reports that five annotators supplied VAD ratings for each sentence and
perspective before its published filtering rule; final annotations average the remaining valid
ratings when at least two remain. The release tables used here contain aggregate references, not a
raw rater-by-item matrix. The paper reports reader-perspective per-rater agreement with the
aggregate reference of Pearson \(r=.738\) for valence and \(r=.595\) for arousal, with MAE `.349`
and `.441`, respectively. These are published dataset-level reliability summaries, not values
recomputed by this project. Run-specific ICC, split-half reliability, rater bootstrap, and human
noise ceilings remain unavailable. See
[Buechel and Hahn (2017), Table 2](https://aclanthology.org/E17-2092.pdf).

The official source tables are monolithic and contain all partition rows. The loader reads the
partition field first, restricts the permitted ID set to `train` and `dev`, and filters text,
references, and metadata to that set before joining them. No test-partition metric, prediction,
identifier hash, or selection result is produced.

## Locked evaluation

- Eligible pool: 9,061 sentences from 136 source documents after excluding one row with missing
  text.
- Outer evaluation: five deterministic source-document `GroupKFold` folds.
- Inner selection: four source-document folds inside each outer-training partition.
- Candidate: TF–IDF/Ridge with alpha selected from `1`, `10`, and `100` using mean valence/arousal
  RMSE and listed-order tie-breaking.
- Comparator: the mean of each outer-training fold; held-out data never determine the mean.
- Coverage: exactly one held-out prediction per eligible item. Predictions remain in memory and
  are not written.
- Uncertainty: 2,000 source-document bootstrap replicates with a 90% valid-replicate threshold.
  Model differences use paired resamples of identical source documents.
- Random seed: `20260724`.

All five outer folds selected alpha `1`. This is a report of the locked procedure, not a general
recommendation for other datasets.

## Aggregate results

| Dimension | Estimator | MAE | RMSE | Pearson | Spearman | CCC |
|---|---|---:|---:|---:|---:|---:|
| Valence | Outer-training mean | .2944 | .4276 | -.0953 | -.0827 | -.0045 |
| Valence | Nested TF–IDF/Ridge | .2839 | .3908 | .4042 | .3862 | .3029 |
| Arousal | Outer-training mean | .2479 | .3307 | -.0393 | -.0415 | -.0008 |
| Arousal | Nested TF–IDF/Ridge | .2508 | .3299 | .2044 | .1837 | .1406 |

Paired source-document bootstrap differences are TF–IDF/Ridge minus the outer-training mean:

| Dimension | Metric difference | Estimate [95% interval] | Interpretation within this benchmark |
|---|---|---:|---|
| Valence | MAE | -.0105 [-.0177, -.0029] | Lower error for TF–IDF/Ridge |
| Valence | RMSE | -.0367 [-.0444, -.0301] | Lower error for TF–IDF/Ridge |
| Valence | CCC | .3075 [.2716, .3447] | Higher concordance for TF–IDF/Ridge |
| Valence | Pearson | .4995 [.3866, .5782] | Higher linear association |
| Valence | Spearman | .4689 [.3619, .5562] | Higher rank association |
| Arousal | MAE | .0029 [-.0012, .0064] | No resolved error advantage |
| Arousal | RMSE | -.0008 [-.0062, .0044] | No resolved error advantage |
| Arousal | CCC | .1414 [.1114, .1706] | Higher concordance for TF–IDF/Ridge |
| Arousal | Pearson | .2437 [.1678, .2904] | Higher linear association |
| Arousal | Spearman | .2251 [.1484, .2792] | Higher rank association |

Every reported paired interval had 2,000 valid replicates. Neither estimator produced values
outside the declared 1–5 scale. Marginal intervals, calibration diagnostics, fold histories,
valid-replicate counts, and source-concentration diagnostics are preserved in the
[aggregate evidence bundle](../evidence/public_text_component_demo_v1/evidence_manifest.json).

The source distribution is concentrated: the largest document contributes about 11.9% of eligible
items and the Kish-style effective source count is about 25.6 despite 136 nominal documents.
Source-grouped evaluation addresses direct document overlap, but the concentration limits
precision and generalization.

## Compute needs

The recorded reference execution used Python 3.12.13 on Windows 11 with CPU-only core
dependencies. It completed in about 131 seconds. The public checkout occupied about 7.6 MB and the
committed aggregate evidence bundle about 102 KB. The Windows peak-memory counter was unavailable,
so the bundle records memory as unavailable rather than inventing a value. No GPU, model-weight
download, or network access during analysis is required.

## Reproduce

From a clean checkout of this repository:

```bash
python -m venv .venv
python -m pip install -r requirements-public-demo-lock.txt
python -m pip install --no-deps -e .
git clone --depth 1 --branch v1.4 \
  https://github.com/JULIELab/EmoBank.git \
  local_public_data/emobank_v1_4
python scripts/run_public_text_component_demo.py \
  --config configs/public_text_component_demo_v1.yaml \
  --emobank-root local_public_data/emobank_v1_4 \
  --output outputs/public_text_component_demo_v1
```

The output directory must not already exist. Validate the resulting aggregate bundle:

```bash
python -c "import json; from pathlib import Path; from text_prosody.evidence_bundle import validate_public_component_demo_bundle; root=Path('outputs/public_text_component_demo_v1'); manifest=json.loads((root/'evidence_manifest.json').read_text()); print(validate_public_component_demo_bundle(root, manifest['artifacts']))"
```

The dataset checkout and new output remain ignored by Git. Do not commit dataset rows, texts,
identifiers, or per-item predictions.

## Claim boundary

This demonstration supports source-grouped performance statements about the two demonstrated text
estimators on the EmoBank v1.4 train/development pool. It does not validate:

- an audio affect component;
- a shared text/audio empirical scale;
- any real model-based text–speech affect discrepancy index;
- masking, incongruence, internal state, or a biomarker;
- DAIC-WOZ or PHQ-8 association; or
- diagnostic, screening, or other clinical use.

Separate text-component evidence cannot establish validity of a difference between text and audio
models.
