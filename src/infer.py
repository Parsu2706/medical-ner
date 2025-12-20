import torch
from typing import Dict, List, Tuple
from huggingface_hub import hf_hub_download

from config import max_len, HF_REPO_ID
from src.model import NERModel
from src.utils import load_tokenizer, load_label_map, get_device


def load_pipeline():
    label2id, id2label = load_label_map()
    tokenizer = load_tokenizer()

    model = NERModel(num_labels=len(label2id),id2label=id2label,label2id=label2id)
    model_path = hf_hub_download(repo_id=HF_REPO_ID,filename="best_model.pt")

    device = get_device()
    model.load(model_path, map_location=device)
    model.to(device)
    model.eval()

    return model, tokenizer, label2id, id2label


def predict(text: str,model,tokenizer,id2label: Dict[int, str]) -> List[Tuple[str, str]]:
    device = get_device()
    words = text.split()
    encodings = tokenizer(words,is_split_into_words=True,return_tensors="pt",truncation=True,max_length=max_len)

    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    output = []
    for token, label_id in zip(tokens, preds):
        if token in tokenizer.all_special_tokens:
            continue
        output.append((token, id2label[label_id]))

    return output
