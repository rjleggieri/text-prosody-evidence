"""Run the locked EmoBank v1.4 reader-perspective component benchmark."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from text_prosody.component_benchmark import (
    PUBLIC_COMPONENT_STATUS,
    content_hash,
    development_objective,
    evaluate_two_target_predictions,
    grouped_bootstrap_intervals,
    load_emobank_reader_release,
    sha256_file,
    verify_test_predictions_once,
)
from text_prosody.test_access import record_test_access
from text_prosody.text_affect import MeanTextVaBaseline, TfidfRidgeTextRegressor


def package_versions(names: list[str]) -> dict[str, str]:
    output = {}
    for name in names:
        try:
            output[name] = version(name)
        except PackageNotFoundError:
            output[name] = "not-installed"
    return output


def load_benchmark_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "extends" not in value:
        return value, {
            "path": str(path),
            "sha256": sha256_file(path),
            "base_path": None,
            "base_sha256": None,
        }
    base_path = path.parent / str(value["extends"])
    observed_base_hash = sha256_file(base_path)
    if observed_base_hash != value["base_config_sha256"]:
        raise RuntimeError("Runtime amendment base-config hash does not match.")
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    roberta = next(
        candidate for candidate in base["candidates"] if candidate["name"] == "roberta_base_va"
    )
    roberta["fixed"]["mixed_precision_dtype"] = value["amendment"]["value"]
    base["status"] = value["status"]
    base["locked_at_utc"] = value["locked_at_utc"]
    base["runtime_amendment"] = value
    return base, {
        "path": str(path),
        "sha256": sha256_file(path),
        "base_path": str(base_path),
        "base_sha256": observed_base_hash,
    }


def git_revision(repository: Path) -> str:
    head = (repository / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    if len(head) == 40 and all(character in "0123456789abcdef" for character in head):
        return head
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_dataset(frame: pd.DataFrame, tokenizer: Any, max_length: int) -> Any:
    import torch

    encoded = tokenizer(
        frame["text"].astype(str).tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = torch.tensor(frame[["valence", "arousal"]].to_numpy(float), dtype=torch.float32)
    return torch.utils.data.TensorDataset(encoded["input_ids"], encoded["attention_mask"], labels)


def transformer_model(model_id: str, revision: str, cache_dir: Path, device: Any) -> Any:
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        num_labels=2,
        problem_type="regression",
        ignore_mismatched_sizes=True,
    )
    return model.to(device)


def transformer_predictions(model: Any, dataset: Any, batch_size: int, device: Any) -> np.ndarray:
    import torch

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    blocks = []
    model.eval()
    with torch.no_grad():
        for input_ids, attention_mask, _labels in loader:
            logits = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            ).logits
            blocks.append(logits.detach().float().cpu().numpy())
    return np.concatenate(blocks, axis=0)


def train_transformer_trial(
    *,
    train: pd.DataFrame,
    dev: pd.DataFrame,
    tokenizer: Any,
    model_id: str,
    revision: str,
    cache_dir: Path,
    checkpoint_dir: Path,
    device: Any,
    fixed: dict[str, Any],
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    import torch
    from transformers import get_linear_schedule_with_warmup

    set_seed(seed)
    model = transformer_model(model_id, revision, cache_dir, device)
    train_data = tensor_dataset(train, tokenizer, int(fixed["max_length"]))
    dev_data = tensor_dataset(dev, tokenizer, int(fixed["max_length"]))
    generator = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=int(fixed["train_batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(fixed["weight_decay"]),
    )
    total_steps = int(fixed["max_epochs"]) * len(train_loader)
    warmup_steps = int(total_steps * float(fixed["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    use_amp = device.type == "cuda" and bool(fixed["mixed_precision_on_cuda"])
    precision_name = str(fixed.get("mixed_precision_dtype", "float16"))
    precision_dtype = torch.bfloat16 if precision_name == "bfloat16" else torch.float16
    if precision_name == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Locked BF16 amendment is unsupported by the CUDA device.")
    use_scaler = use_amp and precision_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    best_objective = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    checkpoint_path = checkpoint_dir / f"roberta_lr_{learning_rate:.0e}.pt"
    started = time.perf_counter()
    for epoch in range(1, int(fixed["max_epochs"]) + 1):
        model.train()
        losses = []
        for input_ids, attention_mask, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=precision_dtype,
                enabled=use_amp,
            ):
                logits = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                ).logits
                loss = torch.nn.functional.mse_loss(logits, labels.to(device))
            if use_scaler:
                previous_scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(fixed["gradient_clip_norm"])
                )
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(fixed["gradient_clip_norm"])
                )
                optimizer.step()
                scheduler.step()
            losses.append(float(loss.detach().cpu()))
        dev_prediction = transformer_predictions(
            model, dev_data, int(fixed["eval_batch_size"]), device
        )
        dev_metrics = evaluate_two_target_predictions(dev, dev_prediction)
        objective = development_objective(dev_metrics)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "development_objective": objective,
                "development_metrics": dev_metrics,
            }
        )
        if objective < best_objective - float(fixed["early_stopping_min_delta"]):
            best_objective = objective
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(fixed["early_stopping_patience"]):
            break
    runtime = time.perf_counter() - started
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "learning_rate": learning_rate,
        "best_epoch": best_epoch,
        "development_objective": best_objective,
        "history": history,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "runtime_seconds": runtime,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--emobank-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    config, config_provenance = load_benchmark_config(config_path)
    if config.get("status") not in {
        "LOCKED_BEFORE_DOWNLOAD",
        "LOCKED_RUNTIME_AMENDMENT_BEFORE_TEST",
    }:
        raise RuntimeError("Text benchmark configuration is not locked.")
    if config["outcome_boundary"]["daic_data_allowed"]:
        raise RuntimeError("Public component benchmark must prohibit DAIC data.")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=False)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir.resolve())

    seed = int(config["reporting"]["random_seed"])
    set_seed(seed)
    frame, data_audit = load_emobank_reader_release(Path(args.emobank_root) / "corpus")
    repository_revision = git_revision(Path(args.emobank_root))
    if not repository_revision.startswith(config["dataset"]["release_commit_prefix"]):
        raise RuntimeError("Local EmoBank checkout does not match the locked release commit.")
    train = frame.loc[frame["split"].eq("train")].reset_index(drop=True)
    dev = frame.loc[frame["split"].eq("dev")].reset_index(drop=True)

    started_mean = time.perf_counter()
    mean_model = MeanTextVaBaseline().fit(train.assign(split="train"))
    mean_dev_prediction = mean_model.predict_batch(dev["text"].astype(str).tolist())
    mean_runtime = time.perf_counter() - started_mean
    mean_dev_metrics = evaluate_two_target_predictions(dev, mean_dev_prediction)

    ridge_config = next(
        candidate for candidate in config["candidates"] if candidate["name"] == "tfidf_ridge"
    )
    ridge_trials: list[dict[str, Any]] = []
    ridge_models: dict[float, TfidfRidgeTextRegressor] = {}
    for alpha in ridge_config["development_grid"]["alpha"]:
        started = time.perf_counter()
        model = TfidfRidgeTextRegressor(
            alpha=float(alpha),
            max_features=int(ridge_config["fixed"]["max_features"]),
        ).fit(train.assign(split="train"))
        prediction = model.predict_batch(dev["text"].astype(str).tolist())
        metrics = evaluate_two_target_predictions(dev, prediction)
        objective = development_objective(metrics)
        ridge_trials.append(
            {
                "alpha": float(alpha),
                "development_objective": objective,
                "development_metrics": metrics,
                "runtime_seconds": time.perf_counter() - started,
                "revision": model.revision_,
            }
        )
        ridge_models[float(alpha)] = model
    selected_ridge = min(
        ridge_trials,
        key=lambda trial: (
            trial["development_objective"],
            ridge_config["development_grid"]["alpha"].index(trial["alpha"]),
        ),
    )
    ridge_model = ridge_models[float(selected_ridge["alpha"])]
    joblib.dump(ridge_model, model_dir / "tfidf_ridge.joblib")

    roberta_config = next(
        candidate for candidate in config["candidates"] if candidate["name"] == "roberta_base_va"
    )
    import torch
    from transformers import AutoTokenizer

    device = torch.device(
        "cuda"
        if config["resources"]["preferred_device"] == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    if device.type == "cpu":
        raise RuntimeError("Locked RoBERTa benchmark requires the available CUDA device.")
    tokenizer = AutoTokenizer.from_pretrained(
        roberta_config["base_model_id"],
        revision=roberta_config["base_model_revision"],
        cache_dir=cache_dir,
    )
    roberta_trials = []
    for learning_rate in roberta_config["development_grid"]["learning_rate"]:
        roberta_trials.append(
            train_transformer_trial(
                train=train,
                dev=dev,
                tokenizer=tokenizer,
                model_id=roberta_config["base_model_id"],
                revision=roberta_config["base_model_revision"],
                cache_dir=cache_dir,
                checkpoint_dir=model_dir,
                device=device,
                fixed=roberta_config["fixed"],
                learning_rate=float(learning_rate),
                seed=seed,
            )
        )
    selected_roberta = min(
        roberta_trials,
        key=lambda trial: (
            trial["development_objective"],
            roberta_config["development_grid"]["learning_rate"].index(trial["learning_rate"]),
        ),
    )

    # The official test partition is evaluated once, only after all development choices are fixed.
    development_history_path = output_dir / "development_histories.json"
    development_history_path.write_text(
        json.dumps(
            {
                "training_mean": {
                    "metrics": mean_dev_metrics,
                    "runtime_seconds": mean_runtime,
                },
                "tfidf_ridge_trials": ridge_trials,
                "roberta_trials": roberta_trials,
                "identical_candidate_seed_set": [seed],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    access_entry = record_test_access(
        output_dir / "test_access_ledger",
        dataset_id=str(config["dataset"]["id"]),
        dataset_version=str(config["dataset"]["release"]),
        split_hash=str(data_audit["split_identifier_sha256"]["test"]),
        run_id=output_dir.name,
        purpose="single_locked_public_component_test_evaluation",
        metadata={
            "evidence_strength": "recorded_once_not_cryptographic_proof",
            "candidate_selection_complete": True,
        },
    )
    test = frame.loc[frame["split"].eq("test")].reset_index(drop=True)
    test_predictions: dict[str, np.ndarray] = {
        "training_mean": mean_model.predict_batch(test["text"].astype(str).tolist()),
        "tfidf_ridge": ridge_model.predict_batch(test["text"].astype(str).tolist()),
    }
    selected_model = transformer_model(
        roberta_config["base_model_id"],
        roberta_config["base_model_revision"],
        cache_dir,
        device,
    )
    selected_model.load_state_dict(
        torch.load(selected_roberta["checkpoint_path"], map_location=device, weights_only=True)
    )
    test_data = tensor_dataset(test, tokenizer, int(roberta_config["fixed"]["max_length"]))
    transformer_started = time.perf_counter()
    test_predictions["roberta_base_va"] = transformer_predictions(
        selected_model,
        test_data,
        int(roberta_config["fixed"]["eval_batch_size"]),
        device,
    )
    transformer_test_runtime = time.perf_counter() - transformer_started
    verify_test_predictions_once(test_predictions, test)

    bootstrap_replicates = int(config["reporting"]["bootstrap_replicates"])
    test_metrics: dict[str, Any] = {}
    prediction_output = test[["item_id", "source_document", "valence", "arousal"]].copy()
    for model_name, prediction in test_predictions.items():
        test_metrics[model_name] = {
            "point_estimates": evaluate_two_target_predictions(test, prediction),
            "grouped_bootstrap_95ci": grouped_bootstrap_intervals(
                test,
                prediction,
                replicates=bootstrap_replicates,
                confidence_level=float(config["reporting"]["bootstrap_confidence_level"]),
                seed=seed,
            ),
        }
        prediction_output[f"{model_name}_valence"] = prediction[:, 0]
        prediction_output[f"{model_name}_arousal"] = prediction[:, 1]
    prediction_path = output_dir / "test_predictions.csv"
    prediction_output.to_csv(prediction_path, index=False)
    split_manifest = {
        split_name: sorted(group["item_id"].astype(str).tolist())
        for split_name, group in frame.groupby("split", sort=True)
    }
    split_path = output_dir / "split_identifiers.json"
    split_path.write_text(json.dumps(split_manifest, indent=2, sort_keys=True) + "\n")

    gpu = {
        "available": bool(torch.cuda.is_available()),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory)
        if torch.cuda.is_available()
        else None,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated())
        if torch.cuda.is_available()
        else None,
    }
    results: dict[str, Any] = {
        "schema_version": 1,
        "status": PUBLIC_COMPONENT_STATUS,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_scope": "EmoBank text-component evidence; UniC retired",
        "daic_data_used": False,
        "daic_outcomes_used": False,
        "test_evaluation_count": 1,
        "config": {
            **config_provenance,
            "locked_at_utc": config["locked_at_utc"].isoformat()
            if hasattr(config["locked_at_utc"], "isoformat")
            else str(config["locked_at_utc"]),
        },
        "dataset": {
            "id": config["dataset"]["id"],
            "release": config["dataset"]["release"],
            "repository_revision": repository_revision,
            "license": config["dataset"]["license"],
            "perspective": "reader",
            "scale": [1.0, 5.0],
            "audit": data_audit,
        },
        "development_selection": {
            "objective": config["selection"]["development_objective"],
            "training_mean": {
                "metrics": mean_dev_metrics,
                "runtime_seconds": mean_runtime,
            },
            "tfidf_ridge_trials": ridge_trials,
            "selected_tfidf_alpha": selected_ridge["alpha"],
            "roberta_trials": roberta_trials,
            "selected_roberta_learning_rate": selected_roberta["learning_rate"],
            "selected_roberta_epoch": selected_roberta["best_epoch"],
        },
        "test_metrics": test_metrics,
        "model_revisions": {
            "training_mean": "training-split-statistic",
            "tfidf_ridge": ridge_model.revision_,
            "roberta_base": roberta_config["base_model_revision"],
            "roberta_finetuned_checkpoint_sha256": selected_roberta["checkpoint_sha256"],
        },
        "runtime": {
            "roberta_test_seconds": transformer_test_runtime,
            "total_roberta_development_seconds": float(
                sum(trial["runtime_seconds"] for trial in roberta_trials)
            ),
        },
        "hardware": {
            "platform": platform.platform(),
            "python": sys.version,
            "cpu_count": os.cpu_count(),
            "gpu": gpu,
        },
        "packages": package_versions(
            [
                "numpy",
                "pandas",
                "scikit-learn",
                "scipy",
                "torch",
                "transformers",
                "tokenizers",
                "safetensors",
                "PyYAML",
            ]
        ),
        "random_seed": seed,
        "development_seed_set": [seed],
        "determinism": {
            "torch_deterministic_algorithms": "enabled_warn_only",
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "remaining_nondeterminism": [
                "GPU kernels may report unsupported deterministic implementations under warn_only.",
                "Hardware, driver, compiler, and library revisions can change "
                "floating-point results.",
                "External model and tokenizer loaders depend on the pinned cached revision.",
            ],
        },
        "artifacts": {
            "test_predictions.csv": sha256_file(prediction_path),
            "split_identifiers.json": sha256_file(split_path),
            "development_histories.json": sha256_file(development_history_path),
            "test_access_ledger_entry": sha256_file(access_entry),
            "tfidf_ridge.joblib": sha256_file(model_dir / "tfidf_ridge.joblib"),
        },
        "interpretation_limits": [
            "Component evidence only; not final DAIC text-model selection.",
            "No UniC paired-modality evidence is available.",
            "Official split source-document overlap is reported as a limitation.",
            "No DAIC data or outcome influenced fitting, selection, or reporting.",
        ],
    }
    results["content_sha256"] = content_hash(results)
    result_path = output_dir / "component_results.json"
    result_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": PUBLIC_COMPONENT_STATUS,
                "result": str(result_path),
                "test_evaluation_count": 1,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
