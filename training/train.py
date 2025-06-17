import os
import torch
import wandb
import evaluate as hug_eval

from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler


def evaluate(model, tokenizer, rouge, epoch, data_loader, config, device):
    model.eval()

    all_preds, all_refs = [], []
    for batch in tqdm(data_loader, desc=f"Validation Epoch {epoch}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        with autocast(config["device"]):
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=config["max_target_length"],
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
            )

        preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        refs = tokenizer.batch_decode(
            torch.where(labels != -100, labels, tokenizer.pad_token_id),
            skip_special_tokens=True,
        )
        rouge.add_batch(predictions=preds, references=refs)
        all_preds.extend(preds)
        all_refs.extend(refs)


def train(
    model,
    tokenizer,
    optimizer,
    scheduler,
    config,
    train_loader,
    eval_loader,
):
    scaler = GradScaler(config["device"])

    accumulation_steps = config.get("gradient_accumulation_steps", 4)

    rouge = hug_eval.load("rouge")
    best_rougeL = 0.0
    patience = config.get("early_stopping_patience", 2)
    patience_counter = 0

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_train_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        for batch_idx, batch in enumerate(progress_bar):
            batch = {
                k: v.to(config["device"], non_blocking=True)
                for k, v in batch.items()
            }

            with autocast(config["device"]):
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            total_train_loss += loss.item() * accumulation_steps

            if (batch_idx + 1) % accumulation_steps == 0 or (
                batch_idx + 1
            ) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            if batch_idx % (15 * accumulation_steps) == 0:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss_step": loss.item() * accumulation_steps,
                    }
                )
                progress_bar.set_postfix(
                    {"loss": loss.item() * accumulation_steps}
                )

        avg_train_loss = total_train_loss / len(train_loader)
        wandb.log({"epoch": epoch, "train_loss": avg_train_loss})
        print(f"Epoch {epoch} Train loss: {avg_train_loss:.4f}")

        rouge = hug_eval.load("rouge")
        evaluate(
            model,
            tokenizer,
            rouge,
            config["epochs"],
            eval_loader,
            config,
            config["device"],
        )

        result = rouge.compute()
        rougeL = result["rougeL"]
        wandb.log(
            {
                "epoch": epoch,
                "validation_rougeL": rougeL,
                "validation_rouge1": result["rouge1"],
                "validation_rouge2": result["rouge2"],
            }
        )
        print(f"Epoch {epoch} Validation ROUGE-L: {rougeL:.4f}")

        if rougeL > best_rougeL:
            best_rougeL = rougeL
            patience_counter = 0
            model.save_pretrained(config["output_dir"])
            tokenizer.save_pretrained(config["output_dir"])

            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(config["output_dir"], "training_state.pt"),
            )

            wandb.run.summary["best_rougeL"] = best_rougeL
            print(f"Saved best model with ROUGE-L: {best_rougeL:.4f}")
        else:
            patience_counter += 1
            print(
                "No improvement in ROUGE-L for"
                f" {patience_counter}/{patience} epochs"
            )
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
