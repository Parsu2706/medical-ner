import os
from dotenv import load_dotenv

load_dotenv()

HF_REPO_ID = os.getenv("HF_REPO_ID")

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
