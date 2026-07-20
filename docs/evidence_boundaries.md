# Claim-to-evidence boundary

| Claim area | Available evidence | Permitted interpretation |
|---|---|---|
| Implementation | Unit, adversarial, static-quality, and clean-install checks | The documented code paths behave as tested. |
| Synthetic pipeline | Deterministic paired synthetic fixture and end-to-end CLI | Software-contract verification only. |
| Text component | Bounded historical EmoBank v1.4 reader-perspective evaluation | Limited evidence about those text estimators under that split and grouping limitation. |
| Audio component | Local runtime smoke verification of a pinned estimator | The adapter can execute and returns shaped finite output; no human-reference validity claim. |
| Benchmark availability | Time-boxed audits found no executable paired dimensional benchmark | Required paired validation is unavailable, not replaced. |
| Cross-modal index | No paired human validation and no demonstrated common empirical scale | No real-data discrepancy index is scientifically validated or frozen. |
| Clinical or internal state | No supporting evidence; related data and outcomes are absent | Unsupported and prohibited. |

## Claims the project may make

- It provides a reference implementation for evidence-aware component evaluation.
- It demonstrates leakage, calibration, reliability, bootstrap, and provenance contracts
  synthetically.
- It reports bounded text-component evidence and an explicit audio-validation limitation.
- It calls any future operational output a model-based text–speech affect discrepancy index.

## Claims the project may not make

- that the difference is human-validated divergence;
- that it measures masking, incongruence, affective truth, or internal state;
- that it is a biomarker, diagnostic, screening tool, or clinical classifier;
- that separate text and audio validity implies validity of their subtraction;
- that synthetic results are empirical validation; or
- that a real-data index or downstream outcome analysis is frozen or authorized.

## Abandoned paired-validation branch

The preferred design requires independent text-only and audio-only human valence/arousal ratings
for the same items, explicit conditions and scales, and leakage-safe grouping metadata. An audited
candidate release lacked that paired dimensional reference. The design is preserved in code and
documentation because it remains the correct evidential requirement, but the unavailable dataset
branch is retired and no restricted audit material is published here.
