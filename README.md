# text-prosody-evidence

[![CI](https://github.com/rjleggieri/text-prosody-evidence/actions/workflows/ci.yml/badge.svg)](https://github.com/rjleggieri/text-prosody-evidence/actions/workflows/ci.yml)
[![Draft paper](https://github.com/rjleggieri/text-prosody-evidence/actions/workflows/paper.yml/badge.svg)](https://github.com/rjleggieri/text-prosody-evidence/actions/workflows/paper.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

`text-prosody-evidence` is an alpha-stage reference implementation for building and auditing
evidence around text and speech affect components. It provides dataset-neutral contracts for
paired human references, leakage-connected grouping, nested grouped evaluation, calibration
sensitivities, reliability interfaces, guarded bootstrap intervals, provenance, and immutable
evidence manifests.

## Evidence boundary

The current evidence is deliberately asymmetric:

- The **text estimator has bounded EmoBank v1.4 component evidence**.
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

## What is implemented

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

## Documentation

- [Statement of need](docs/statement_of_need.md)
- [Evidence and claim boundaries](docs/evidence_boundaries.md)
- [Text component evidence](docs/text_component_evidence.md)
- [API overview](docs/api.md)
- [Development setup](docs/development.md)
- [Roadmap](ROADMAP.md)
- [JOSS draft](paper/paper.md)

This public repository launched on **2026-07-20** from a sanitized export of prior private research
development. Its public history is intentionally honest and does not claim that private work was
public incubation. See [development history](docs/development_history.md).

## Contributing and support

Please read [CONTRIBUTING.md](CONTRIBUTING.md), [GOVERNANCE.md](GOVERNANCE.md), and
[SUPPORT.md](SUPPORT.md). Report security or privacy concerns through the process in
[SECURITY.md](SECURITY.md).

## License

Code and original documentation are available under the [MIT License](LICENSE). Dataset licenses,
model licenses, and third-party terms remain separate; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
