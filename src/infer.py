from typing import Dict, List, Tuple

import torch

from config import max_len, best_model_path
from src.model import NERModel
from src.utils import (
    load_tokenizer,
    load_label_map,
    get_device,
)


def load_pipeline():
    label2id, id2label = load_label_map()
    tokenizer = load_tokenizer()
    model = NERModel(num_labels=len(label2id), id2label=id2label, label2id=label2id)
    device = get_device()
    model.load(best_model_path, map_location=device)
    model.to(device)
    model.eval()
    return model, tokenizer, label2id, id2label


def predict(text: str,model,tokenizer,id2label: Dict[int, str],max_len_override: int = None) -> List[Tuple[str, str]]:
    device = get_device()
    words = text.split()
    effective_max_len = max_len if max_len_override is None else max_len_override

    encodings = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=effective_max_len,
    )

    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)[0].cpu().tolist()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    output_tokens: List[str] = []
    output_labels: List[str] = []
    current_token = ""
    current_label = "O"

    for token, label_id in zip(tokens, predictions):
        if token in tokenizer.all_special_tokens:
            continue
        label = id2label.get(label_id, "O")
        if token.startswith("##"):
            current_token += token[2:]
        else:
            if current_token:
                output_tokens.append(current_token)
                output_labels.append(current_label)
            current_token = token
            current_label = label

    if current_token:
        output_tokens.append(current_token)
        output_labels.append(current_label)

    return list(zip(output_tokens, output_labels))
