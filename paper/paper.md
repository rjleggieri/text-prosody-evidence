---
title: "text-prosody-evidence: evidence-aware reference methods for affect components"
tags:
  - Python
  - affective computing
  - multimodal evaluation
  - reproducible research
authors:
  - name: Robert Leggieri
    affiliation: 1
affiliations:
  - name: University of Texas at Austin, United States
    index: 1
date: 20 July 2026
bibliography: paper.bib
---

# Summary

`text-prosody-evidence` is a Python reference implementation for evaluating text and speech affect
components without silently converting software output into a validated psychological construct.
It provides contracts for modality-specific human references, transitive leakage grouping, nested
grouped evaluation, calibration sensitivities, reliability reporting, guarded bootstrap intervals,
and immutable evidence manifests. A deterministic synthetic command exercises the full contract
path without requiring a dataset, model checkpoint, network connection, or item-level output.

The alpha release deliberately exposes an evidence gap. Text estimators have bounded component
evidence from EmoBank [@buechel2017emobank], while the pinned audio adapter has runtime verification
only. No available benchmark established independent text-only and audio-only dimensional human
ratings on the same items with sufficient grouping metadata. The software therefore does not claim
that independently estimated text and audio coordinates share an empirical scale, and it does not
freeze a scientifically valid real-data difference score.

# Statement of need

Subtracting independently trained affect estimators is computationally simple but scientifically
demanding. Calibration can mix dimensions or leak held-out information; related source items can
cross nominal folds; reliability uncertainty can be confused with source-sampling uncertainty; and
overlapping marginal intervals do not answer a paired model-comparison question. General-purpose
machine-learning libraries such as scikit-learn [@pedregosa2011scikit] provide essential primitives,
but they do not encode this evidence contract as a coherent, fail-closed workflow.

The package addresses these risks with six linked mechanisms. First, a paired-reference schema
requires explicit text-only and audio-only annotation conditions, homogeneous scales, and composite
dataset/version/item keys. Second, connected components close every available source-video,
speaker, creator/channel, duplicate-content, and provenance relation before splitting. Third,
outer grouped folds estimate the complete selection procedure while inner folds select only the
prespecified text candidate; the audio estimator remains fixed. Fourth, dimension-wise affine
calibration is primary, with unclipped predictions, out-of-range diagnostics, and parallel
prespecified sensitivities. Fifth, rater and source-group uncertainty are kept separate, and
bootstrap intervals are withheld unless at least 90% of attempted replicates are valid. Finally,
artifact roles and SHA-256 values are validated without regenerating missing evidence.

# Software verification and evidence boundary

Unit and adversarial tests cover leakage closure, outer-fold isolation, identical candidate seeds,
dimension-preserving calibration, paired bootstrap differences, valid-replicate thresholds,
reliability fallbacks, path safety, immutable ledgers, and import side effects. The command-line
demonstration generates synthetic modality references and estimator outputs, runs grouped held-out
calibration, and writes aggregate metrics and a manifest. Synthetic results verify implementation
contracts only.

The historical text-component table uses reader-perspective EmoBank ratings and reports continuous
metrics including concordance correlation [@lin1989concordance]. Its published split contains
substantial source-document overlap, so the repository labels that result bounded rather than a
source-independent generalization estimate. A separate executable demonstration uses nested
source-document folds on the published train/development pool, excludes the test partition, and
publishes an aggregate hash-validated evidence bundle. This improves component evidence for the
demonstrated text estimators only. The optional audio smoke configuration references a pinned
speech-emotion estimator in the family discussed by @wagner2023dawn, but runtime success is not
treated as human-reference validity.

Until paired human evidence is obtained, the only permitted operational term is “model-based
text–speech affect discrepancy index.” The project does not claim a biomarker, diagnosis, masking,
incongruence, internal state, or human-validated divergence. Separate component evidence does not
establish validity of their difference.

# Availability and development

The package is released under the MIT License. Core installation supports Python 3.10–3.12 and has
no data or model-weight dependency. Optional benchmark dependencies and upstream datasets retain
their own terms. The public repository began on 2026-07-20 from a sanitized export of prior private
development; its public issue and release history is intentionally not backdated. Contributions are
invited for portability testing, documentation, API stabilization, and scrutiny or reproduction of
the license-compatible public text-component demonstration without expanding unsupported claims.

# Acknowledgements

AI-assisted coding and documentation tools contributed suggestions during development. The author
reviewed the released code and documentation and accepts responsibility for their content. No
restricted payloads or human-subject records are distributed with the package.

# References
