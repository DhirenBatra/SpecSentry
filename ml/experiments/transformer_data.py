"""Encode frozen PROMISE_exp splits for DistilBERT."""

import random
from statistics import median

from ml.experiments.classification_data import TASK_LABELS, target_labels

MODEL_NAME = "distilbert-base-uncased"
MODEL_REPOSITORY = "distilbert/distilbert-base-uncased"
MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
MAX_LENGTH = 128


def label_ids(task):
    return {label: index for index, label in enumerate(TASK_LABELS[task])}


def decode_labels(ids, task):
    names = {index: label for label, index in label_ids(task).items()}
    return [names[index] for index in ids]


def prepare_split(rows, tokenizer, task):
    """Keep row order and tokenize clean_text with the pretrained vocabulary."""
    texts = [row.clean_text for row in rows]
    encoded = tokenizer(texts, truncation=True, max_length=MAX_LENGTH, return_token_type_ids=False)
    labels = label_ids(task)
    return [
        {"input_ids": ids, "attention_mask": mask, "labels": labels[label]}
        for ids, mask, label in zip(
            encoded["input_ids"], encoded["attention_mask"], target_labels(rows, task), strict=True
        )
    ]


def token_length_summary(rows, tokenizer):
    inputs = tokenizer(
        [row.clean_text for row in rows], truncation=False, return_attention_mask=False, verbose=False
    )
    lengths = sorted(len(ids) for ids in inputs["input_ids"])
    return {
        "median_tokens": median(lengths),
        "maximum_tokens": lengths[-1],
        "truncated_examples": sum(length > MAX_LENGTH for length in lengths),
    }


def batch_indices(examples, batch_size, shuffle_seed=None):
    """Pad within batches of similar length. Shuffle short pools during fitting."""
    indices = list(range(len(examples)))
    rng = None if shuffle_seed is None else random.Random(shuffle_seed)
    if rng:
        rng.shuffle(indices)
    pool_size = batch_size * 8 if rng else len(indices)
    batches = []
    for start in range(0, len(indices), pool_size):
        pool = indices[start:start + pool_size]
        pool.sort(key=lambda index: len(examples[index]["input_ids"]))
        batches.extend(pool[offset:offset + batch_size] for offset in range(0, len(pool), batch_size))
    if rng:
        rng.shuffle(batches)
    return batches
