# Maintainer guide: what this project is and why the rebuild stopped where it did

This guide is for the researcher maintaining `text-prosody-evidence`. It explains the research
story, the software, and the limits of the evidence without assuming research-software expertise.
The shorter public claim boundary remains authoritative in
[evidence_boundaries.md](evidence_boundaries.md).

## If you remember only five things

1. The original research question asked whether text-derived and speech-derived affect disagree in
   DAIC-WOZ interviews and whether that model-based disagreement is associated with PHQ-8.
2. The legacy school project did run DAIC/PHQ analyses and produced historical results. Those
   results are privately preserved, quarantined, and not publication evidence.
3. The rebuild found that subtracting two model outputs is not automatically a valid measurement.
   The required paired human validation and common empirical scale are currently unavailable.
4. The public software implements evaluation safeguards and verifies them synthetically. It does
   not validate a real text–speech discrepancy construct.
5. The current publication target is the methods/software and evidence framework. Any new DAIC or
   PHQ work in the rebuilt publication pipeline remains conditional future research.

## The research question that started the project

The original question was about DAIC-WOZ interviews and PHQ-8 depression scores:

> Can text-derived affect and speech-derived affect differ within an interview, and is that
> model-based difference associated with PHQ-8?

The intended workflow was roughly: estimate valence and arousal from transcript text, estimate the
same dimensions from speech, compare the two estimates, and then explore whether that comparison is
associated with PHQ-8. The DAIC/PHQ part was meant to be exploratory, not a diagnostic system.

The legacy school project did run DAIC/PHQ analyses and produced historical results. Those results
are privately preserved and quarantined because they are not publication evidence. They are not
included in this public repository and are not used to support its claims.

The rebuilt publication pipeline started from the same broad question but required stronger
measurement validation. It stopped before running any **new** DAIC affect extraction, PHQ
association, prediction, threshold selection, or clinical modeling because the proposed comparison
was not yet a validated measurement. The original question remains scientifically interesting, but
this public repository does not answer it.

## Why subtraction became a measurement problem

Suppose a text model reports valence `0.7` and an audio model reports valence `0.2`. Subtracting
them gives `0.5`, but that arithmetic alone does not show that `0.5` has a stable human meaning.
The models may have been trained on different populations, annotation instructions, scales, and
domains. One model's `0.7` may not be comparable with the other's `0.2`.

There are three separate questions:

1. Does the text estimator track human text-only valence and arousal?
2. Does the audio estimator track human audio-only valence and arousal?
3. After suitable calibration, does the difference between those estimates track the difference
   between paired human text-only and audio-only judgments for the same items?

Evidence for questions 1 and 2 does not automatically answer question 3. Errors can be correlated,
calibration can distort dimensions, and independently evaluated components may not share a common
empirical scale. This is why the current project calls any future operational quantity a
**model-based text–speech affect discrepancy index**, not a human-validated divergence measure.

PHQ-8 creates another risk. If models, calibration, features, or thresholds are selected because
they predict PHQ-8 well, the project is no longer independently testing an affect measurement. It
is optimizing a predictor of the outcome and then describing that predictor as a validated
measurement. The repository therefore rejects clinical-outcome columns from calibration and
selection interfaces.

## Why UniC was investigated

The required benchmark needed the same content rated under separate conditions:

- humans see text only and rate valence and arousal;
- humans hear audio only and rate valence and arousal;
- the two records can be linked to the same item;
- the rating scale and annotation instructions are explicit; and
- related items can be kept together during evaluation.

UniC was investigated because its research description discussed separate text, audio, visual,
and multimodal conditions. On paper, that made it a plausible candidate for evaluating the exact
paired quantity rather than merely evaluating two components on unrelated datasets.

The findings came from a private, privacy-safe aggregate audit of the gated release. Restricted
payloads and item-level information are not public. The public repository records the aggregate
determination and its scientific consequence, but it does not make the gated audit independently
reproducible for readers without authorized access to the same release.

The investigation was a schema and feasibility audit, not a model benchmark. No UniC affect-model
inference, calibration, candidate selection, or discrepancy evaluation was run during that audit.

## Why the audited UniC release failed the evidence contract

This conclusion applies specifically to the release audited at repository revision
`ac2c172a97f41343dd18f71bd2218528f87883d6`. It is not a claim about every past or future UniC
release.

The audited full payload was a pre-split package of aligned text, audio, and vision representations
for 964 items. It had stable item linkage, but the audited payload did not contain the required:

- text-only human valence and arousal targets;
- audio-only human valence and arousal targets;
- explicit text-only and audio-only condition identifiers;
- dimensional scale definitions or anchors;
- rater records, rater counts, or aggregation rules; or
- source, speaker, creator/channel, duplicate-content, or provenance grouping fields.

A separately audited small release object contained categorical sentiment information. Sentiment
classes are not interchangeable with two-dimensional valence/arousal ratings. Precomputed feature
tensors are model inputs or representations, not human reference judgments. Neither was accepted
as a substitute.

The result was
`BLOCKED_REQUIRED_PAIRED_DIMENSIONAL_ANNOTATIONS_ABSENT`. The UniC branch was retired for this
specific paired dimensional benchmark. The preferred paired-validation design was preserved as an
auditable abandoned branch rather than rewritten after seeing what the release contained.

## Why DAIC/PHQ modeling was stopped in the rebuilt pipeline

The legacy school project had already run DAIC/PHQ analyses. The rebuild did not treat those
historical results as publication evidence. Instead, it stopped before running any new DAIC/PHQ
analysis for scientific reasons, not because the Python pipeline was impossible to run.

- The text and audio outputs had not been shown to share a common empirical scale.
- The audio component had no executed independent human-reference benchmark.
- The difference between components had no paired human validation.
- Using PHQ-8 to choose models or calibration would risk outcome-guided measurement construction.
- Continuing data authorization and governance must be confirmed separately whenever a future
  DAIC phase is proposed.

The current decision is therefore: no **new** DAIC affect extraction, PHQ association, prediction,
threshold selection, or clinical classification in the rebuilt publication workflow. A later
researcher decision cannot create the missing scientific evidence by itself. The quarantined
legacy analyses remain historical context, not an exception that authorizes reuse as publication
evidence.

## Keep the five project states separate

| State | Meaning in this project | Example |
|---|---|---|
| Implemented software behavior | Code does what a testable contract says it should do. | Related source items are kept in one fold. |
| Synthetic verification | The implementation succeeds and fails appropriately on invented data with known structure. | The CLI runs grouped calibration and suppresses invalid intervals on synthetic fixtures. |
| Real empirical evidence | A real human-reference evaluation was actually run and its limitations are reported. | The bounded historical EmoBank text-component results. |
| Proposed future research | A scientifically conditional path that has not been executed in the rebuilt workflow or authorized by this guide. | Reconnecting externally frozen components to a new exploratory DAIC/PHQ analysis. |
| Abandoned research branch | A documented design that was stopped because its activation condition failed. | Using the exact audited UniC release as the paired dimensional benchmark. |

Moving from one row to another requires new evidence. A software test does not become an empirical
result, and researcher approval does not turn missing validation into validation.

## What the public software does

The software is a reference implementation of safeguards for a future, qualifying benchmark. It:

- checks that paired human references have the required modality conditions, dimensions, scales,
  keys, and grouping metadata;
- joins every available leakage relationship into transitive connected components;
- evaluates a selection procedure with outer held-out grouped folds and inner grouped selection;
- keeps the audio estimator fixed while a prespecified text candidate may be selected;
- uses dimension-wise affine calibration as primary, so valence predicts valence and arousal
  predicts arousal;
- reports cross-dimensional and robust mappings only as parallel sensitivities;
- reports unclipped primary predictions, a clipped sensitivity, and out-of-range diagnostics;
- separates rater uncertainty from source-group uncertainty;
- withholds a bootstrap interval when fewer than 90% of attempted replicates are valid;
- compares models through paired resampling of the same source groups;
- records hashes, configuration information, and required evidence files without fabricating
  missing artifacts; and
- provides deterministic synthetic fixtures and a CLI demonstration.

These are implemented software behaviors covered by unit and adversarial tests. They are not proof
that a particular psychological construct is valid.

## What the software does not do

The public package does not:

- contain or process DAIC-WOZ or PHQ-8 data as part of its normal package workflow;
- contain the gated UniC payload, participant records, raw interview media, or real item-level
  predictions;
- provide a scientifically frozen real-data discrepancy index;
- establish audio component validity;
- make text and audio scales comparable merely by running calibration code;
- diagnose depression or any other condition;
- infer masking, incongruence, affective truth, or internal state; or
- automatically approve a dataset because it has enough rows or groups for a mathematical split.

The optional model adapters and historical benchmark scripts are tools, not bundled model weights
or scientific results. Invoking a model successfully shows runtime behavior, not validity.

## Who might realistically use it

Likely users are:

- affective-computing researchers planning separate text-only and audio-only validation;
- multimodal ML researchers who need leakage-safe grouped evaluation;
- benchmark designers deciding which metadata and rater information to release;
- research-software authors who want fail-closed evidence artifacts; and
- students learning why a computed difference is not automatically a validated construct.

It is not intended for clinicians, patient screening, real-time mental-health decisions, or anyone
seeking a ready-made depression model.

## What JOSS means here

JOSS is the *Journal of Open Source Software*. A JOSS submission is centered on useful, open,
maintained research software rather than a conventional paper built around a new empirical result.
Reviewers inspect the repository as well as a short paper: installation, tests, documentation,
statement of need, contribution process, licensing, and evidence that the project is a genuine
research-software contribution all matter.

Creating a public repository and a draft JOSS paper begins an incubation and review-preparation
process; it does not make the project publishable automatically. Public use, issue-based
development, comparison with related software, reviewer judgment, and continued maintenance remain
open. Nothing in this repository predicts or guarantees JOSS submission or acceptance.

## Major package components

| Component | Plain-English purpose | Current evidence status |
|---|---|---|
| `contracts` | Provides shared validation rules and error types, including checks that prohibited identifiers or groups do not overlap between training and held-out data. | Implemented and tested. |
| `external_benchmark` | Defines the paired human-reference schema and builds transitive leakage components. | Implemented and tested with synthetic fixtures. |
| `calibration` | Maps model valence/arousal onto a declared human scale using primary and sensitivity mappings. | Implemented; no real cross-modal calibration is scientifically frozen. |
| `external_evaluation` | Runs grouped held-out evaluation, nested candidate selection, feasibility reports, sensitivities, and bootstrap comparisons. | Implemented and tested synthetically. |
| `reliability` | Computes rater-level reliability/noise-ceiling summaries when raw ratings exist and reports a clear fallback when they do not. | Implemented and tested synthetically. |
| `discrepancy` | Computes signed valence/arousal differences and L1, L2, or cosine summaries after a scale check. | Arithmetic is implemented; real-data construct validity is absent. |
| `text_affect` | Defines text-estimator interfaces, EmoBank normalization, and simple text baselines. | Bounded historical EmoBank component evidence exists. |
| `audio_affect` | Defines the pinned audio adapter, output ordering, batching, quality flags, and resume ledger. | Runtime verification only. |
| `component_benchmark` | Supplies continuous metrics, grouped bootstrap guards, split checks, and EmoBank benchmark helpers. | Implemented; historical aggregate results are documented separately. |
| `metrics` | Computes regression metrics and requires a training-fold mean baseline. | Implemented and tested. |
| `evidence_bundle` | Verifies that every required evidence artifact exists at a safe relative path and matches its SHA-256. | Implemented and tested; it never creates missing evidence. |
| `provenance` | Creates hashes, package-version summaries, and machine-readable run metadata. | Implemented and tested through dependent modules. |
| `test_access` | Writes non-overwriting test-access records and states that “recorded once” is not cryptographic proof. | Implemented and tested. |
| `reproducibility` | Applies consistent Python, NumPy, and optional Torch seed/determinism settings. | Implemented; some hardware/library nondeterminism can remain. |
| `config` and `cli` | Load safe YAML configuration and run the aggregate-only synthetic demonstration. | Implemented and clean-install tested. |

See [api.md](api.md) for the shorter technical map.

## What happens when a dataset passes the evidence contract

Passing the schema check is the beginning of an evaluation, not automatic scientific approval.
A qualifying dataset would follow this sequence:

```text
Candidate release
    |
    v
License, schema, scale, condition, key, and rater audit
    |
    v
Build transitive leakage components from every available relation
    |
    v
Researcher reviews group concentration and fold feasibility
    |
    v
Outer grouped folds estimate the complete selection procedure
    |
    +--> Inner grouped folds select the text candidate only
    |       using locked candidates, seeds, objective, and tie-break
    |
    +--> Fixed audio estimator + primary calibration
    |
    v
Exactly one held-out prediction per item
    |
    v
Component metrics, discrepancy metrics, reliability, guarded bootstraps,
parallel sensitivities, out-of-range diagnostics, and hashed evidence bundle
    |
    v
Researcher decides what the evidence supports; no automatic construct approval
```

After outer-fold performance estimates the complete procedure, the final-candidate interface may
repeat the same locked inner-style rule on the complete external benchmark and fit calibration only
after candidate selection. Outer-test metrics, sensitivity results, DAIC, and PHQ are not allowed to
choose that candidate.

### Example of an acceptable paired benchmark

Imagine a release in which 2,000 public clips have:

- independent text-only human valence and arousal means;
- independent audio-only human valence and arousal means;
- the same declared 1-to-5 anchors for both conditions;
- stable dataset/version/item keys;
- raw rater IDs and ratings, or clearly documented aggregate-only status;
- source-video, speaker, channel, and duplicate-family identifiers; and
- terms permitting the intended research and aggregate publication.

If those fields are complete, internally consistent, sufficiently reliable, and form enough
well-distributed connected components for the prespecified folds, the dataset can proceed to the
locked grouped evaluation. It still does not guarantee that the resulting index will validate;
the held-out results decide what evidence was obtained.

## What happens when a dataset fails the evidence contract

The workflow fails closed:

- missing required fields or mixed scales cause validation errors;
- duplicate composite keys or blank grouping values are rejected;
- absent leakage relations prevent a defensible grouped analysis;
- insufficient connected components produce a feasibility report, not automatic approval;
- aggregate-only ratings trigger an explicit reliability/noise-ceiling limitation;
- degenerate bootstrap samples are counted by reason, and intervals disappear below the 90%
  valid-replicate threshold; and
- missing or hash-mismatched evidence files cause bundle verification to fail rather than being
  recreated.

The researcher may document the negative result, find a genuinely different qualifying release,
or narrow the research question. The researcher should not rename unsuitable labels or use model
features as human ground truth.

### Rejection example: unsuitable labels

The exact audited UniC release is the real project example. Its audited payload did not provide the
four paired dimensional human targets and condition metadata. Its categorical sentiment object and
precomputed representations could support different questions, but they could not be renamed into
text-only/audio-only valence and arousal. It was therefore rejected for this estimand.

### Rejection example: inadequate grouping metadata

Imagine a dataset with excellent paired text-only and audio-only valence/arousal ratings, but only
an item ID. Several clips may come from the same speaker, video, channel, or duplicated source, yet
none of those relationships is available. Randomly assigning items to folds could put nearly
identical or speaker-related material in both training and test. The ratings are suitable, but the
dataset is rejected for the planned generalization claim because leakage-safe evaluation cannot be
constructed.

## Why synthetic examples are included

Synthetic examples answer: “Does the software enforce the intended rules?” They make it possible
to test known relationships, deliberate leakage, degenerate bootstrap samples, and failure cases
without exposing restricted data or waiting for a rare benchmark.

Synthetic examples do **not** answer: “Does this construct work on real people or real affect?” A
test can prove that outer folds were held out or that an interval was suppressed correctly. It
cannot prove that text and audio models share a psychologically meaningful scale.

## What real empirical evidence exists

### Text component

There is bounded historical evidence from the EmoBank v1.4 reader-perspective benchmark. The
documented candidates included a mean baseline, TF-IDF/Ridge, and a pinned RoBERTa regression
candidate. Aggregate performance is preserved in
[text_component_evidence.md](text_component_evidence.md).

The result is deliberately called bounded because source documents overlap substantially across
the published train, development, and test splits. The historical evidence bundle also predates the
current public evidence specification. It supports limited statements about those text estimators
under that evaluation; it does not select a final DAIC model or validate a cross-modal difference.

### Audio component

The pinned audio adapter has runtime verification: it loads when explicitly invoked, observes the
documented output order, enforces sampling-rate and batching contracts, returns finite shaped
outputs on generated signals, and supports a resume ledger. No executed independent audio-only
human valence/arousal benchmark establishes its component validity. Runtime success must not be
reported as accuracy, reliability, or human-reference validity.

### Cross-modal quantity

No real benchmark has shown both components to be valid on a shared empirical scale and no paired
human benchmark has validated their difference. Therefore no real-data discrepancy index is
scientifically validated or frozen. The public cross-modal pipeline is a reference implementation
demonstrated synthetically.

## How the project could eventually reconnect to DAIC and PHQ

This is proposed future research, not an authorized or active analysis.

1. Obtain a qualifying paired benchmark, or otherwise obtain evidence that directly supports the
   intended real-data quantity.
2. Audit its license, annotation conditions, scales, raters, grouping, reliability, and release
   revision.
3. Predefine and approve the candidate set, fixed audio estimator, grouping relations, folds,
   objective, tie-break, calibration, sensitivities, bootstrap settings, and claims.
4. Run the external benchmark without DAIC or PHQ influence and freeze only what the evidence
   supports.
5. Reconfirm DAIC authorization, privacy controls, and the exact exploratory analysis plan.
6. Apply the already frozen components and transformations to DAIC without outcome-driven tuning.
7. Analyze PHQ association as exploratory and conditional, with no diagnostic or internal-state
   claim.

If step 1 never becomes possible, the software paper remains separate and new DAIC/PHQ modeling in
the rebuilt publication workflow should remain stopped. The quarantined legacy results do not fill
that evidence gap.

## The maintainer's personal responsibilities

The maintainer is responsible for understanding and approving:

- the difference between software correctness, synthetic verification, component evidence, and
  construct validity;
- why separate text and audio evidence does not validate their subtraction;
- every dataset's license, access terms, redistribution limits, consent implications, and allowed
  publication outputs;
- whether grouping metadata support the intended generalization claim, not merely the requested
  number of folds;
- any change to scientific defaults, candidates, seeds, objectives, tie-breaks, calibration,
  sensitivities, thresholds, or evidence requirements;
- whether manuscript and README claims match the evidence actually produced;
- privacy-safe handling of restricted data, predictions, parameters, identifiers, and artifacts;
- authorship, AI-assistance disclosure, citations, releases, issue responses, and maintenance; and
- any future decision to reactivate DAIC or PHQ work in the rebuilt publication pipeline.

Automated tests can help enforce a decision after it is made. They cannot decide whether the
scientific question, license, ethics, or claim is appropriate.

## Five-minute explanation for another graduate student

“I originally wanted to compare affect estimated from DAIC transcript text with affect estimated
from speech and then explore whether that difference related to PHQ-8. The legacy school project
did run DAIC/PHQ analyses, but those historical results are privately quarantined and are not
publication evidence. The rebuild therefore returned to the measurement question. The hard part
was not the subtraction: the two models may use different scales and may make correlated errors, so
their difference is not automatically a valid measurement.

“I looked for a benchmark with the same items rated by humans in text-only and audio-only
conditions for both valence and arousal. UniC looked promising from its description, but the exact
release I audited contained aligned multimodal representations without the required paired
dimensional human targets, scale, rater, or leakage-grouping fields. I did not reinterpret sentiment
labels or features as the missing annotations.

“So the rebuilt pipeline stopped before running any new DAIC affect extraction or PHQ analysis.
The public project is now research software that encodes the evaluation safeguards we would need
if a suitable benchmark becomes available: strict schema checks, connected grouping to prevent
leakage, nested held-out evaluation, dimension-preserving calibration, reliability interfaces,
guarded bootstraps, and hashed evidence manifests.

“The text side has bounded historical EmoBank evidence. The audio side has runtime verification
only. The complete cross-modal pipeline works on deterministic synthetic fixtures, which proves
software behavior but not scientific validity. There is no validated real-data discrepancy index,
and the software is not clinical. The contribution is making the evidence requirements and stop
rules explicit.”

## Questions a JOSS reviewer may ask

### What problem does this software solve?

It makes evidence requirements and evaluation safeguards explicit for studies comparing text and
speech affect estimators. It separates software behavior from scientific validity.

### Why is this more than a collection of utility functions?

The modules implement one connected fail-closed workflow: paired-reference validation, transitive
grouping, nested grouped selection, constrained calibration, reliability, guarded inference, and
immutable evidence verification.

### Where is the real-world validation?

The text component has bounded historical EmoBank evidence. Audio validity and the real cross-modal
quantity remain unvalidated. The repository states this as a limitation rather than substituting
synthetic results.

### Did the project ever run DAIC or PHQ analyses?

Yes. The legacy school project ran historical DAIC/PHQ analyses, but those results are privately
quarantined and are not publication evidence. No new DAIC affect extraction or PHQ analysis has
been run in the rebuilt publication workflow.

### Why include a synthetic demonstration?

It gives users a reproducible way to verify software contracts without restricted data. It is
labeled software verification, not empirical evidence.

### Can I reproduce the package without private data?

Yes. Core installation, tests, and the synthetic CLI require no private dataset or model weights.
Optional upstream datasets and models retain separate terms.

### How does the software prevent leakage?

It connects items transitively through every available source, speaker, creator/channel,
duplicate-content, and provenance relation, then keeps each resulting component within one fold.

### How is model selection kept separate from evaluation?

Inner grouped folds select the text candidate using locked rules. Outer grouped test folds estimate
the complete procedure and cannot influence selection. The audio estimator remains fixed.

### What happens when bootstrap samples are invalid?

The software reports attempted, valid, and reason-specific invalid counts. It withholds an interval
unless at least 90% of replicates are valid.

### Is the project ready for JOSS acceptance?

No acceptance claim is made. The repository has installation, tests, documentation, governance,
an open issue roadmap, and a draft paper, but reviewer assessment, community use, software need,
and submission readiness remain open questions.

### Who governs scientific changes?

The maintainer records and reviews changes to evidence requirements and defaults. Scientific
changes require an issue, adversarial tests, documented evidence consequences, and an explicit
decision under [GOVERNANCE.md](../GOVERNANCE.md).

## Statements the maintainer should not make

Do not say that:

- the project has validated text–speech affect divergence;
- the current discrepancy index measures a human internal state;
- the index detects masking, incongruence, deception, or affective truth;
- the index is a biomarker, diagnostic aid, screener, or clinical classifier;
- the audio estimator has been validated against independent human audio-only V/A ratings;
- successful runtime inference demonstrates model accuracy;
- EmoBank text evidence validates the audio component or the difference between components;
- synthetic results are real empirical validation;
- calibration automatically makes independently trained models commensurate;
- UniC in general lacks dimensional annotations; the conclusion is limited to the exact audited
  release;
- categorical sentiment or supplied features are equivalent to paired human valence/arousal;
- no DAIC/PHQ analysis ever occurred; the legacy school analyses did occur but are quarantined;
- the quarantined legacy DAIC/PHQ results are publication evidence;
- the rebuilt publication workflow has run new DAIC affect extraction, PHQ association,
  prediction, threshold selection, or clinical modeling;
- this package authorizes or scientifically freezes any DAIC or PHQ analysis;
- “recorded once” is cryptographic proof that a test partition was never accessed elsewhere;
- a mathematically feasible fold count is scientific approval; or
- JOSS submission, review, or acceptance is likely, promised, or guaranteed.

## Glossary

1. **Affect dimensions:** Continuous coordinates such as valence (negative to positive) and arousal
   (low to high activation).
2. **Component estimator:** One model that estimates affect from one modality, such as text or
   audio.
3. **Paired human reference:** Human ratings for the same item under separate text-only and
   audio-only conditions.
4. **Annotation condition:** What information raters were allowed to see or hear when rating.
5. **Evidence contract:** The required fields, metadata, and quality conditions a benchmark must
   satisfy before an analysis is allowed.
6. **Commensurate scales:** Scales whose values have enough shared meaning to be compared or
   subtracted.
7. **Calibration:** A mapping from model outputs to a declared human-reference scale.
8. **Leakage:** Information or closely related content appearing on both the training and held-out
   sides of an evaluation.
9. **Connected component:** All items linked directly or transitively by any available leakage
   relation.
10. **Grouped fold:** A data split that keeps every item in the same group or connected component
    on one side.
11. **Nested grouped evaluation:** Inner grouped folds make development choices; separate outer
    grouped folds estimate the complete procedure.
12. **Bootstrap:** Repeated resampling used to describe uncertainty; here, whole source groups or
    raters are resampled as appropriate.
13. **Validity guard:** A rule that stops or suppresses an analysis when its assumptions are not
    met.
14. **Evidence manifest:** A machine-readable list of required artifacts and hashes used to verify
    what a run actually produced.
15. **Model-based text–speech affect discrepancy index:** A computed difference between calibrated
    text and speech model outputs; currently a software-defined quantity, not a validated human
    construct.
