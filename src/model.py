from torch import nn
from transformers import AutoModelForTokenClassification
from peft import LoraConfig, get_peft_model, TaskType

from config import pretrained_model


class NERModel(nn.Module):
    def __init__(self, num_labels: int, id2label, label2id):
        super().__init__()
        base_model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.TOKEN_CLS,
        )
        self.model = get_peft_model(base_model, lora_config)

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def save(self, path: str):
        import torch
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location=None):
        import torch
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)
        self.eval()
