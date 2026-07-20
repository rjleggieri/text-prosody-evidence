# Statement of need

Cross-modal affect studies often subtract outputs from independently trained text and audio
estimators. That operation is easy to code but difficult to justify: estimator scales may not be
commensurate, source relationships can leak across folds, calibration can absorb held-out
information, and marginal confidence intervals do not test paired model differences.

`text-prosody-evidence` makes those assumptions inspectable. It supplies fail-closed contracts for
paired modality-specific human references; transitive leakage grouping; nested grouped selection;
dimension-preserving calibration with parallel sensitivities; reliability and noise-ceiling
interfaces; guarded source and rater bootstraps; and immutable, hash-addressed evidence manifests.

The package is useful even when the desired benchmark is unavailable: it distinguishes missing
scientific evidence from missing software behavior and supports deterministic synthetic contract
verification without promoting that verification into a validity claim. This release intentionally
demonstrates that boundary. It does not offer a clinical measure or a validated real-data index.
