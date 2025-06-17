import torch

from datasets import load_dataset
from torch.utils.data import DataLoader


def make_collate_fn(tokenizer):
    def collate_fn(batch):
        input_ids = torch.tensor([ex["input_ids"] for ex in batch])
        attention_mask = torch.tensor([ex["attention_mask"] for ex in batch])
        labels = torch.tensor([ex["labels"] for ex in batch])
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return collate_fn


def tokenize_batch(examples, tokenizer, max_input_len, max_target_len):
    model_inputs = tokenizer(
        examples["article"],
        max_length=max_input_len,
        truncation=True,
        padding="max_length",
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["highlights"],
            max_length=max_target_len,
            truncation=True,
            padding="max_length",
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def prepare_data(
    tokenizer, config, dataset_name="cnn_dailymail", revision="3.0.0"
):
    dataset = load_dataset(dataset_name, revision)

    train_ds = dataset["train"].select(range(100_000))
    val_ds = dataset["validation"].select(range(10_000))

    train_ds = train_ds.map(
        lambda x: tokenize_batch(
            x,
            tokenizer,
            config["max_input_length"],
            config["max_target_length"],
        ),
        batched=True,
        remove_columns=["article", "highlights", "id"],
        num_proc=4,
    )

    val_ds = val_ds.map(
        lambda x: tokenize_batch(
            x,
            tokenizer,
            config["max_input_length"],
            config["max_target_length"],
        ),
        batched=True,
        remove_columns=["article", "highlights", "id"],
        num_proc=4,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer),
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer),
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader
