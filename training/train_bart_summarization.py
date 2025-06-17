import os
import torch
import wandb
import pandas as pd

from datasets import load_dataset
from transformers import (
    BartForConditionalGeneration,
    BartTokenizerFast,
    AdamW,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from transformers import AdamW
from torch.utils.data import DataLoader, Dataset

from train import train
from utils import load_config
from data_utils import *


def main():
    config = load_config("config.yaml")

    print(config)

    os.makedirs(config["output_dir"], exist_ok=True)

    wandb.init(project="bart-summarization", config=config)

    tokenizer = BartTokenizerFast.from_pretrained(config["model_name"])
    model = BartForConditionalGeneration.from_pretrained(config["model_name"])
    model.to(config["device"])

    train_loader, val_loader = prepare_data(tokenizer, config)

    optimizer = AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    total_steps = len(train_loader) * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=total_steps,
    )

    train(
        model, tokenizer, optimizer, scheduler, config, train_loader, val_loader
    )


if __name__ == "__main__":
    main()
