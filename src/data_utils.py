from typing import Dict, List, Tuple
import random
from collections import defaultdict

from datasets import load_dataset
from transformers import AutoTokenizer

from config import dataset_name, pretrained_model, max_len


def load_splits_and_labels(test_ratio: float = 0.1,val_ratio: float = 0.1,seed: int = 42,max_samples_per_class: int = 4000):
    dataset = load_dataset(dataset_name)
    train_split = dataset["train"]
    label2id = {
        "O": 0,
        "B-PROBLEM": 1,
        "I-PROBLEM": 2,
        "E-PROBLEM": 3,
        "S-PROBLEM": 4,
        "B-TREATMENT": 5,
        "I-TREATMENT": 6,
        "E-TREATMENT": 7,
        "S-TREATMENT": 8,
        "B-TEST": 9,
        "I-TEST": 10,
        "E-TEST": 11,
        "S-TEST": 12,
    }
    id2label = {v: k for k, v in label2id.items()}

    all_tokens = train_split["sentence"]
    all_tags = train_split["tags"]

    problem_label_ids = {label2id["B-PROBLEM"],label2id["I-PROBLEM"],label2id["E-PROBLEM"],label2id["S-PROBLEM"],}
    test_label_ids = {label2id["B-TEST"],label2id["I-TEST"],label2id["E-TEST"],label2id["S-TEST"],}
    treatment_label_ids = {label2id["B-TREATMENT"],label2id["I-TREATMENT"],label2id["E-TREATMENT"],label2id["S-TREATMENT"],}

    random.seed(seed)

    label_group_to_indices = defaultdict(list)

    for i, tag_sequence in enumerate(all_tags):
        tag_set = set(tag_sequence)
        if tag_set & problem_label_ids:
            label_group_to_indices["problem"].append(i)
        if tag_set & test_label_ids:
            label_group_to_indices["test"].append(i)
        if tag_set & treatment_label_ids:
            label_group_to_indices["treatment"].append(i)

    samples_per_group = min(
        len(label_group_to_indices["problem"]),
        len(label_group_to_indices["test"]),
        len(label_group_to_indices["treatment"]),
        max_samples_per_class,
    )

    selected_indices = set()

    for group_name in ("problem", "test", "treatment"):
        selected_indices.update(
            random.sample(label_group_to_indices[group_name], samples_per_group)
        )

    selected_indices = list(selected_indices)
    random.shuffle(selected_indices)

    selected_tokens = [all_tokens[i] for i in selected_indices]
    selected_tags = [all_tags[i] for i in selected_indices]

    n = len(selected_tokens)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_test - n_val

    train_tokens = selected_tokens[:n_train]
    train_tags = selected_tags[:n_train]

    val_tokens = selected_tokens[n_train:n_train + n_val]
    val_tags = selected_tags[n_train:n_train + n_val]

    test_tokens = selected_tokens[n_train + n_val:]
    test_tags = selected_tags[n_train + n_val:]

    splits = {
        "train": {"tokens": train_tokens, "tags": train_tags},
        "validation": {"tokens": val_tokens, "tags": val_tags},
        "test": {"tokens": test_tokens, "tags": test_tags},
    }

    return splits, label2id, id2label

def get_tokenizer():
    return AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)

def align_labels(tokens: List[str],label_ids: List[int],tokenizer,max_len_override: int = None,) -> Tuple[Dict[str, List[int]], List[int]]:
    assert len(tokens) == len(label_ids), "Tokens and labels must be same length"

    effective_max_len = max_len if max_len_override is None else max_len_override

    encodings = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=effective_max_len,
        return_attention_mask=True,
    )

    word_ids = encodings.word_ids()
    aligned_labels: List[int] = []

    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(-100)
            continue
        label_id = label_ids[word_idx]
        label_name = None

        if word_idx != previous_word_idx:
            aligned_labels.append(label_id)
        else:
            label_name = label_name or id_to_label(label_id)
            if label_name.startswith(("B-", "E-", "S-")):
                label_name = "I-" + label_name.split("-", 1)[1]
                aligned_labels.append(label_to_id(label_name))
            else:
                aligned_labels.append(label_id)

        previous_word_idx = word_idx

    return encodings, aligned_labels

def id_to_label(label_id: int) -> str:
    reverse_map = {
        0: "O",
        1: "B-PROBLEM",
        2: "I-PROBLEM",
        3: "E-PROBLEM",
        4: "S-PROBLEM",
        5: "B-TREATMENT",
        6: "I-TREATMENT",
        7: "E-TREATMENT",
        8: "S-TREATMENT",
        9: "B-TEST",
        10: "I-TEST",
        11: "E-TEST",
        12: "S-TEST",
    }
    return reverse_map[label_id]

def label_to_id(label_name: str) -> int:
    label2id = {
        "O": 0,
        "B-PROBLEM": 1,
        "I-PROBLEM": 2,
        "E-PROBLEM": 3,
        "S-PROBLEM": 4,
        "B-TREATMENT": 5,
        "I-TREATMENT": 6,
        "E-TREATMENT": 7,
        "S-TREATMENT": 8,
        "B-TEST": 9,
        "I-TEST": 10,
        "E-TEST": 11,
        "S-TEST": 12,
    }
    return label2id[label_name]
