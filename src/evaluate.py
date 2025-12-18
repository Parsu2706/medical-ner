from typing import List, Tuple, Dict
import torch
from torch.utils.data import DataLoader
from seqeval.metrics import classification_report, f1_score
from src.utils import get_device, load_label_map

def collect_predictions(model,data_loader: DataLoader,ignore_index: int = -100,) -> Tuple[List[List[str]], List[List[str]]]:
    device = get_device()
    _, id2label = load_label_map()
    model.to(device)
    model.eval()

    true_sequences: List[List[str]] = []
    pred_sequences: List[List[str]] = []

    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()

            for true_labels, pred_labels in zip(labels, predictions):
                true_seq = []
                pred_seq = []
                for true_label, pred_label in zip(true_labels, pred_labels):
                    if true_label == ignore_index:
                        continue
                    true_seq.append(id2label[int(true_label)])
                    pred_seq.append(id2label[int(pred_label)])
                true_sequences.append(true_seq)
                pred_sequences.append(pred_seq)

    return true_sequences, pred_sequences

def evaluate(model, data_loader: DataLoader) -> Dict[str, float]:
    true_labels, pred_labels = collect_predictions(model, data_loader)
    print("=== Evaluation ===")
    print(classification_report(true_labels, pred_labels, digits=4))
    f1_score_value = f1_score(true_labels, pred_labels)
    print(f"Micro F1: {f1_score_value:.4f}")
    return {"micro_f1": f1_score_value}
