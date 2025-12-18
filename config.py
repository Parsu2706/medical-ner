import os

base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, "models")
os.makedirs(model_dir, exist_ok=True)

dataset_name = "HUMADEX/english_ner_dataset"
pretrained_model = "emilyalsentzer/Bio_ClinicalBERT"
max_len = 128
train_batch_size = 16
val_batch_size = 32
epochs = 5
lr = 3e-5
weight_decay = 0.01
warmup_ratio = 0.1
grad_clip = 1.0
log_every = 50
best_model_path = os.path.join(model_dir, "best_model.pt")
tokenizer_path = model_dir
label_map_path = os.path.join(model_dir, "label_map.json")
