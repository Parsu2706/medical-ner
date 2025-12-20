import json
import torch
import random
import numpy as np
from typing import Dict, List
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

from config import HF_REPO_ID, pretrained_model


def load_label_map():
    path = hf_hub_download(repo_id=HF_REPO_ID,filename="label_map.json")
    with open(path, "r") as f:
        data = json.load(f)

    label2id = data["label2id"]
    id2label = {int(k): v for k, v in data["id2label"].items()}
    return label2id, id2label


def load_tokenizer():
    return AutoTokenizer.from_pretrained(HF_REPO_ID or pretrained_model)


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
