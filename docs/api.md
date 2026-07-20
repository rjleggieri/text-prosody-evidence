# API overview

The alpha API is organized by evidence responsibility.

## Calibration

`text_prosody.calibration` provides dimension-wise affine calibration as the primary mapping.
Valence is mapped from model valence only and arousal from model arousal only. The
cross-dimensional affine mapping is explicitly sensitivity-only; robust location-scale mapping is
another prespecified sensitivity.

## Human-reference contracts and grouping

`text_prosody.external_benchmark` validates homogeneous scales and explicit `text_only` and
`audio_only` conditions. Composite keys use dataset, version, and item. Connected components close
all available source-video, speaker, creator/channel, duplicate-content, and provenance relations.

## Evaluation

`text_prosody.external_evaluation` includes grouped held-out calibration, nested grouped candidate
selection, final candidate-freeze interfaces, unclipped primary predictions, clipped sensitivity,
out-of-range diagnostics, fold-feasibility summaries, source-group bootstrap intervals, and paired
model-comparison intervals. A mathematical fold minimum never grants scientific approval.

The word `freeze` in a function name identifies immutable selection behavior. It does not imply
that this release freezes or validates a real-data cross-modal construct.

## Reliability

`text_prosody.reliability` separates rater-level ICC/split-half and rater bootstrap from source-group
resampling. When only aggregate ratings exist, the interface reports that rater-level reliability
and human noise ceilings are unavailable rather than estimating them from model output.

## Evidence and provenance

`text_prosody.evidence_bundle` verifies a declared artifact set and hashes without fabricating or
regenerating missing evidence. `text_prosody.test_access` is an append-only, non-overwriting access
ledger. `text_prosody.provenance` provides canonical JSON and SHA-256 helpers.
`text_prosody.reproducibility` centralizes optional deterministic Torch settings so scripts and
tests use the same contract.

All interfaces are alpha and may change before 1.0. Scientific-default changes follow the process
in `GOVERNANCE.md`.
