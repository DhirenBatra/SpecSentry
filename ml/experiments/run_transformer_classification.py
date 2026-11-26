"""Fine-tune DistilBERT on the frozen Phase 2 partitions and compare with Phase 3."""

import gc
import json
import os
import platform
import random
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import torch
from sklearn.metrics import f1_score
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from ml.evaluation.classification_metrics import evaluate_predictions
from ml.evaluation.transformer_results import (
    PHASE3_COMMIT, PHASE3_SHA256, compare_to_baseline, format_errors,
    load_baselines, write_transformer_figures,
)
from ml.experiments.classification_data import (
    PHASE2_COMMIT, SPLIT_INFO, TASK_LABELS, ensure_disjoint, read_split, target_labels,
)
from ml.experiments.run_classification_baselines import write_errors, write_json
from ml.experiments.transformer_data import (
    MAX_LENGTH, MODEL_NAME, MODEL_REPOSITORY, MODEL_REVISION, batch_indices,
    decode_labels, label_ids, prepare_split, token_length_summary,
)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "ml" / "artifacts" / "transformer_classification"
CACHE_DIR = ARTIFACTS / "hf_cache"
RESULTS_DIR = ROOT / "research" / "results"
RESULTS_PATH = RESULTS_DIR / "transformer_classification_results.json"
ERRORS_PATH = RESULTS_DIR / "transformer_classification_test_errors.csv"

SEED = 42
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
FROZEN_BLOCKS = 4
WARMUP_FRACTION = 0.10


def pretrained_kwargs(local_files_only=False):
    return {
        "revision": MODEL_REVISION,
        "cache_dir": CACHE_DIR,
        "local_files_only": local_files_only,
        "token": False,
    }


def load_tokenizer(local_files_only=False):
    return AutoTokenizer.from_pretrained(MODEL_REPOSITORY, **pretrained_kwargs(local_files_only))


def select_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return torch.device("cuda")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    return torch.device("cpu")


def build_model(task, device):
    labels = label_ids(task)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_REPOSITORY,
        use_safetensors=True,
        num_labels=len(labels),
        label2id=labels,
        id2label={index: name for name, index in labels.items()},
        **pretrained_kwargs(),
    )
    model.distilbert.embeddings.requires_grad_(False)
    for layer in model.distilbert.transformer.layer[:FROZEN_BLOCKS]:
        layer.requires_grad_(False)
    return model.to(device)


def class_weights(samples, task, device):
    if task == "binary":
        return None
    counts = Counter(item["labels"] for item in samples)
    # Square-root weighting moderates the inverse-frequency ratio on small classes.
    weights = [1 / counts[index] ** 0.5 for index in range(len(label_ids(task)))]
    average = sum(counts[index] * weight for index, weight in enumerate(weights)) / len(samples)
    return torch.tensor([weight / average for weight in weights], dtype=torch.float32, device=device)


def create_batch(samples, indices, collator, device):
    batch = collator([samples[index] for index in indices])
    labels = batch.pop("labels").to(device)
    return {key: values.to(device) for key, values in batch.items()}, labels


@torch.inference_mode()
def predict(model, samples, collator, task, device):
    model.eval()
    predictions = [None] * len(samples)
    for indices in batch_indices(samples, BATCH_SIZE):
        inputs, _ = create_batch(samples, indices, collator, device)
        prediction_ids = model(**inputs).logits.argmax(dim=1).tolist()
        for index, predicted in zip(indices, prediction_ids, strict=True):
            predictions[index] = predicted
    return decode_labels(predictions, task)


def train_epoch(model, samples, batches, collator, device, optimizer, scheduler, weights, task, epoch):
    model.train()
    # Frozen features have fixed dropout behavior throughout training.
    model.distilbert.embeddings.eval()
    for layer in model.distilbert.transformer.layer[:FROZEN_BLOCKS]:
        layer.eval()
    loss_total = 0.0
    loss_weight = 0.0
    for step, indices in enumerate(batches, start=1):
        optimizer.zero_grad(set_to_none=True)
        inputs, labels = create_batch(samples, indices, collator, device)
        logits = model(**inputs).logits
        loss = F.cross_entropy(logits, labels, weight=weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        weight = len(indices) if weights is None else weights[labels].sum().item()
        loss_total += loss.item() * weight
        loss_weight += weight
        if step == 1 or step % 10 == 0 or step == len(batches):
            print(f"{task} epoch {epoch}/{EPOCHS} · batch {step}/{len(batches)}", flush=True)
    return loss_total / loss_weight


def fit_task(task, train, validation, tokenizer, device):
    random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    training_data = prepare_split(train, tokenizer, task)
    validation_data = prepare_split(validation, tokenizer, task)
    validation_labels = target_labels(validation, task)
    collator = DataCollatorWithPadding(tokenizer)
    model = build_model(task, device)
    trainable = {name: param for name, param in model.named_parameters() if param.requires_grad}
    optimizer = torch.optim.AdamW(trainable.values(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    steps = len(batch_indices(training_data, BATCH_SIZE, SEED)) * EPOCHS
    warmup = max(1, int(steps * WARMUP_FRACTION))

    def learning_rate_multiplier(step):
        if step < warmup:
            return step / warmup
        return max(0.0, (steps - step) / (steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)
    weights = class_weights(training_data, task, device)
    history = []
    best_score = -1.0
    best_epoch = None
    best_state = None
    for epoch in range(1, EPOCHS + 1):
        start = perf_counter()
        batches = batch_indices(training_data, BATCH_SIZE, SEED + epoch)
        train_loss = train_epoch(
            model, training_data, batches, collator, device, optimizer, scheduler, weights, task, epoch,
        )
        predictions = predict(model, validation_data, collator, task, device)
        score = float(f1_score(
            validation_labels, predictions, labels=list(TASK_LABELS[task]), average="macro", zero_division=0,
        ))
        history.append({"epoch": epoch, "training_loss": train_loss, "validation_macro_f1": score})
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {name: param.detach().cpu().clone() for name, param in trainable.items()}
        print(
            f"{task} epoch {epoch} · loss {train_loss:.4f} · validation macro F1 {score:.4f}"
            f" · {perf_counter() - start:.0f}s",
            flush=True,
        )

    with torch.no_grad():
        for name, param in trainable.items():
            param.copy_(best_state[name])
    destination = ARTIFACTS / task
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    print(f"{task} selected epoch {best_epoch} and saved model/tokenizer", flush=True)
    labels = label_ids(task)
    return {
        "pretrained_model": MODEL_NAME,
        "pretrained_revision": MODEL_REVISION,
        "task": task,
        "label_to_id": labels,
        "label_names": TASK_LABELS[task],
        "training_count": len(train),
        "seed": SEED,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "cpu_threads": torch.get_num_threads() if device.type == "cpu" else None,
        "epochs": EPOCHS,
        "selected_epoch": best_epoch,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "optimizer": "AdamW",
        "weight_decay": WEIGHT_DECAY,
        "schedule": {"type": "linear decay after warmup", "warmup_steps": warmup, "total_steps": steps},
        "gradient_clip_norm": 1.0,
        "frozen_embeddings": True,
        "frozen_transformer_blocks": FROZEN_BLOCKS,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable.values()),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "class_weighting": "sqrt inverse frequency on training labels" if weights is not None else "unweighted",
        "class_weights": {label: weights[index].item() for label, index in labels.items()} if weights is not None else None,
        "validation_macro_f1": best_score,
        "training_history": history,
        "local_model_directory": destination.relative_to(ROOT).as_posix(),
    }


def run():
    baselines = load_baselines(RESULTS_DIR)
    train = read_split("train")
    validation = read_split("validation")
    ensure_disjoint({"train": train, "validation": validation})
    device = select_device()
    print(f"Training on {device} · {torch.get_num_threads()} CPU threads", flush=True)
    tokenizer = load_tokenizer()
    word_lengths = json.loads((RESULTS_DIR / "dataset_statistics.json").read_text(encoding="utf-8"))
    token_lengths = {
        "training": token_length_summary(train, tokenizer),
        "validation": token_length_summary(validation, tokenizer),
    }
    print(f"WordPiece lengths at max_length={MAX_LENGTH}: {token_lengths}", flush=True)

    tasks = {}
    for task in TASK_LABELS:
        tasks[task] = fit_task(task, train, validation, tokenizer, device)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Both checkpoints are fixed before the held-out text is opened.
    test = read_split("test")
    ensure_disjoint({"train": train, "validation": validation, "test": test})
    token_lengths["test"] = token_length_summary(test, tokenizer)
    errors = []
    for task, metadata in tasks.items():
        destination = ROOT / metadata["local_model_directory"]
        saved_tokenizer = AutoTokenizer.from_pretrained(destination, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            destination, local_files_only=True, use_safetensors=True,
        ).to(device)
        test_data = prepare_split(test, saved_tokenizer, task)
        predicted = predict(model, test_data, DataCollatorWithPadding(saved_tokenizer), task, device)
        actual = target_labels(test, task)
        evaluated = evaluate_predictions(actual, predicted, TASK_LABELS[task])
        metadata["final_test_metrics"] = evaluated["test_metrics"]
        write_json(destination / "metadata.json", metadata)
        metadata.update(evaluated)
        metadata["comparison"] = compare_to_baseline(task, evaluated, baselines[task])
        errors.extend(format_errors(test, actual, predicted, task))
        print(f"{task} held-out metrics: {evaluated['test_metrics']}", flush=True)
        del model

    figures = write_transformer_figures(RESULTS_DIR, tasks, baselines, len(test))
    versions = {
        name: version(name) for name in
        ("torch", "transformers", "huggingface-hub", "tokenizers", "safetensors", "scikit-learn", "matplotlib")
    }
    write_json(RESULTS_PATH, {
        "dataset": {
            "name": "PROMISE_exp", "phase2_commit": PHASE2_COMMIT, "splits": SPLIT_INFO,
            "test_projects": len({row.project_id for row in test}),
            "portability_test_examples": sum(row.original_label == "PO" for row in test),
            "cleaned_word_lengths": word_lengths["processed"]["word_lengths"],
            "wordpiece_lengths": token_lengths,
        },
        "baseline": {"phase3_commit": PHASE3_COMMIT, "source_sha256": PHASE3_SHA256},
        "selection": {"metric": "validation_macro_f1", "ties": "earliest epoch"},
        "versions": {"python": platform.python_version(), **versions},
        "tasks": tasks,
        "error_records": ERRORS_PATH.relative_to(ROOT).as_posix(),
        "figures": figures,
    })
    write_errors(ERRORS_PATH, errors)
    print(f"Saved results, {len(errors)} error rows, and {len(figures)} figures.", flush=True)


if __name__ == "__main__":
    run()
