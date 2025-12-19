import json
import os
from typing import Dict, List

from huggingface_hub import hf_hub_download
from config import (pretrained_model,HF_REPO_ID,HF_LABEL_MAP_FILE,HF_TOKENIZER_REPO)
import torch
import random
import numpy as np
from transformers import AutoTokenizer

from config import label_map_path, tokenizer_path, pretrained_model

def save_label_map(label2id: Dict[str, int], id2label: Dict[int, str]):
    data = {
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
    }
    with open(label_map_path, "w") as file:
        json.dump(data, file, indent=2)

def load_label_map():
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_LABEL_MAP_FILE,
    )
    with open(path, "r") as file:
        data = json.load(file)

    label2id = data["label2id"]
    id2label = {int(k): v for k, v in data["id2label"].items()}
    return label2id, id2label

def save_tokenizer(tokenizer: AutoTokenizer):
    tokenizer.save_pretrained(tokenizer_path)

def load_tokenizer():
    return AutoTokenizer.from_pretrained(
        HF_TOKENIZER_REPO or pretrained_model
    )
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def decode_predictions(input_ids: torch.Tensor,pred_ids: torch.Tensor,tokenizer,id2label: Dict[int, str]) -> List[List[str]]:
    result = []
    for seq_ids, seq_labels in zip(input_ids, pred_ids):
        tokens = tokenizer.convert_ids_to_tokens(seq_ids)
        labels = [id2label.get(int(label), "O") for label in seq_labels]
        result.append(list(zip(tokens, labels)))
    return result

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
