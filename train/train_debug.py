import os
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


global RANK

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen-7B")

@dataclass
class TrainingArguments(TrainingArguments):
    use_lora : bool = field(default=False)

@dataclass
class DataArguments:
    data_path : str = field(default=None)




class AudioDataset(Dataset):
    def __init__(self, args) -> None:
        super().__init__()
        self.data_path = args.data_path


    def __len__(self):
        pass

    def __getitem__(self, index):
        return super().__getitem__(index)