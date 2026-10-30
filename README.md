# text-prosody-evidence

[![CI](https://github.com/rjleggieri/text-prosody-evidence/actions/workflows/ci.yml/badge.svg)](https://github.com/rjleggieri/text-prosody-evidence/actions/workflows/ci.yml)
[![Draft paper](https://github.com/rjleggieri/text-prosody-evidence/actions/workflows/paper.yml/badge.svg)](https://github.com/rjleggieri/text-prosody-evidence/actions/workflows/paper.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

`text-prosody-evidence` is Python evaluation software for text and speech affect research.
I built grouping, nested evaluation, calibration, model-comparison, and provenance workflows
so that model results can be traced to their data, splits, and evaluation settings.

## What I built

- Grouping of related records across source documents, speakers, creators, duplicate content,
  and shared provenance.
- Nested grouped evaluation that separates model selection from held-out assessment.
- Calibration and reliability interfaces, including explicitly separated sensitivity analyses.
- Paired source-group bootstrap comparisons for prediction error, correlation, and concordance.
- Aggregate evidence bundles with immutable manifests, source metadata, and test-access records.

## Measured result: EmoBank text-component benchmark

On **9,061 sentences from 136 source documents**, nested TF-IDF/Ridge reduced valence
prediction error compared with an outer-training-mean baseline under document-grouped evaluation.

| Text estimator | Valence MAE | Valence RMSE |
|---|---:|---:|
| Outer-training mean baseline | 0.2944 | 0.4276 |
| Nested TF-IDF/Ridge | **0.2839** | **0.3908** |

The benchmark used five outer document folds, four inner selection folds, and 2,000 paired
source-document bootstrap replicates. The paired valence RMSE difference was -0.0367,
with a 95% interval of [-0.0444, -0.0301].

These are text-component results on the EmoBank v1.4 train/development pool.
[Evaluation details and reproduction instructions](docs/public_text_component_demo.md)
and the [aggregate evidence bundle](evidence/public_text_component_demo_v1/evidence_manifest.json)
document the run.

## Evaluation features

- strict paired-reference scale, condition, and composite-key contracts;
- transitive connected components across source, speaker, creator/channel, duplicate-content, and
  provenance relations;
- grouped nested evaluation and a separate final-candidate selection interface;
- dimension-wise affine primary calibration, with cross-dimensional affine and robust alternatives
  retained only as parallel sensitivities;
- unclipped primary predictions plus clipped sensitivity and out-of-range diagnostics;
- source-group and rater bootstraps kept distinct, with a 90% valid-replicate threshold;
- paired source-group model comparisons for MAE, RMSE, CCC, Pearson, and Spearman metrics;
- reliability/noise-ceiling interfaces and aggregate-only fallbacks;
- fail-closed evidence-bundle and test-access ledgers.

## Install

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

Python 3.10–3.12 is supported. See [installation](docs/installation.md) for platform details.

## Synthetic quick start

```bash
text-prosody-evidence synthetic-reference --output outputs/synthetic-demo
```

The command creates aggregate synthetic metrics, an explicitly synthetic calibration record, and
a hash manifest. It never writes item-level predictions. See [quick start](docs/quickstart.md).

## Documentation

- [Statement of need](docs/statement_of_need.md)
- [Evidence and claim boundaries](docs/evidence_boundaries.md)
- [Text component evidence](docs/text_component_evidence.md)
- [Public grouped text-component demonstration](docs/public_text_component_demo.md)
- [API overview](docs/api.md)
- [Development setup](docs/development.md)
- [Roadmap](ROADMAP.md)
- [JOSS draft](paper/paper.md)

The public repository launched on **2026-07-20** from a sanitized export of earlier private
research development. See [development history](docs/development_history.md).

## Evidence boundary

This is an alpha-stage research software reference implementation.

The current evidence is deliberately asymmetric:

- The **text estimator has bounded EmoBank v1.4 component evidence**, including a reproducible
  source-document-grouped train/development demonstration. The published test partition was not
  reused for this demonstration.
- The **audio estimator has runtime verification only**, not independent human-reference validity
  evidence.
- The **cross-modal pipeline is verified only with deterministic synthetic data**. This is software
  verification, not scientific validation.
- No scientifically valid real-data discrepancy index has been frozen. The only permitted name for
  a future output at present is a **model-based text–speech affect discrepancy index**.
- No DAIC-WOZ data, PHQ outcomes, participant data, gated UniC payloads, real item predictions, or
  restricted model weights are included or required by the package.
- The software is nonclinical. It does not measure internal state, masking, incongruence, a
  biomarker, or a diagnostic construct.

Separate component evidence does **not** establish validity of the difference between components.
The original paired-human-validation design remains documented as an unavailable evidence branch;
it has not been silently replaced by sentiment labels, multimodal labels, or model features.

## Contributing and support

Please read [CONTRIBUTING.md](CONTRIBUTING.md), [GOVERNANCE.md](GOVERNANCE.md), and
[SUPPORT.md](SUPPORT.md). Report security or privacy concerns through the process in
[SECURITY.md](SECURITY.md).

## License

Code and original documentation are available under the [MIT License](LICENSE). Dataset licenses,
model licenses, and third-party terms remain separate; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
