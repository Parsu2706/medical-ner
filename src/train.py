import math
import os

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import mlflow

from config import (max_len,train_batch_size,val_batch_size,epochs,lr,weight_decay,warmup_ratio,grad_clip,best_model_path,log_every,model_dir,dataset_name,pretrained_model)
from src.data_utils import (load_splits_and_labels,get_tokenizer,align_labels)
from src.dataset import NERDataset
from src.model import NERModel
from src.utils import (save_tokenizer,save_label_map,get_device,set_seed)
from src.evaluate import evaluate

def build_data_loaders():
    data_splits, label2id, id2label = load_splits_and_labels()
    tokenizer = get_tokenizer()

    def encode_split(split_name: str):
        tokens = data_splits[split_name]["tokens"]
        tags = data_splits[split_name]["tags"]
        encodings = {"input_ids": [], "attention_mask": []}
        all_labels = []
        for token_list, tag_list in zip(tokens, tags):
            encoding, aligned_labels = align_labels(token_list, tag_list, tokenizer, max_len)
            encodings["input_ids"].append(encoding["input_ids"])
            encodings["attention_mask"].append(encoding["attention_mask"])
            all_labels.append(aligned_labels)
        return encodings, all_labels

    train_encodings, train_labels = encode_split("train")
    val_encodings, val_labels = encode_split("validation")

    train_dataset = NERDataset(train_encodings, train_labels)
    val_dataset = NERDataset(val_encodings, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    save_tokenizer(tokenizer)
    save_label_map(label2id, id2label)

    return train_loader, val_loader, label2id, id2label

def train():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, label2id, id2label = build_data_loaders()

    num_labels = len(label2id)
    model = NERModel(num_labels=num_labels, id2label=id2label, label2id=label2id)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(warmup_ratio * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    mlflow.set_experiment("medical_ner_humadex_lora")
    best_f1 = -math.inf
    global_step = 0

    with mlflow.start_run():
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("pretrained_model", pretrained_model)
        mlflow.log_param("max_len", max_len)
        mlflow.log_param("train_batch_size", train_batch_size)
        mlflow.log_param("val_batch_size", val_batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("lr", lr)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("warmup_ratio", warmup_ratio)
        mlflow.log_param("grad_clip", grad_clip)

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

            for batch in progress_bar:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                global_step += 1

                if global_step % log_every == 0:
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                    mlflow.log_metric("train_loss", loss.item(), step=global_step)

            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch + 1} loss: {avg_epoch_loss:.4f}")
            mlflow.log_metric("epoch_loss", avg_epoch_loss, step=epoch + 1)

            metrics = evaluate(model, val_loader)
            f1_score = metrics["micro_f1"]
            mlflow.log_metric("val_f1", f1_score, step=epoch + 1)

            if f1_score > best_f1:
                best_f1 = f1_score
                print(f"New best F1: {best_f1:.4f}, saving model")
                model.save(best_model_path)

        mlflow.log_metric("best_val_f1", best_f1)
        if os.path.exists(best_model_path):
            mlflow.log_artifact(best_model_path, artifact_path="model")
        if os.path.exists(model_dir):
            mlflow.log_artifacts(model_dir, artifact_path="artifacts")

    print(f"Done. Best F1: {best_f1:.4f}")

if __name__ == "__main__":
    train()
