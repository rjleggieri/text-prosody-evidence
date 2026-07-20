# Synthetic quick start

Run the reference demonstration after installation:

```bash
text-prosody-evidence synthetic-reference \
  --output outputs/synthetic-demo \
  --seed 20260720 \
  --bootstrap-replicates 200
```

The output directory must not already exist. It contains:

- `metrics.json`: aggregate synthetic metrics and guarded grouped-bootstrap intervals;
- `reference_calibration.json`: parameters fitted to synthetic values, explicitly marked as
  non-scientific; and
- `run_manifest.json`: seed, status, and SHA-256 values for the two artifacts.

No item-level predictions are persisted. Re-running with the same seed reproduces numerical
content, apart from filesystem metadata. These outputs verify code contracts only. They are not
evidence that text and audio model outputs share a valid empirical scale.
