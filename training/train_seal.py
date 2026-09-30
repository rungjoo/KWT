import sys
import torch
import json
import argparse
import wandb
from datasets import Dataset
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import CKPT_DIR, get_model_name, resolve_threshold, knowledge_file, checkpoint_dir

######################################
# 1) Data loading
######################################

def load_data(filepath):
    """
    Load and prepare training data.

    Args:
        filepath: Path to the JSON file containing training data

    Returns:
        list: List of processed data items
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data_type = []
    for result in data['results']:
        question = result['question']
        right_answer = result['right_answer']

        item = {
            'question': question,
            'response': right_answer,
        }

        data_type.append(item)

    return data_type

######################################
# 2) Prompt / tokenization
######################################
def create_prompt(question):
    return f"Question: {question}\n\nAnswer:"

def preprocess_tokenize(item, tokenizer, max_length=1024):
    question = item['question']
    prompt = create_prompt(question)

    full_text = prompt + " " + item['response'] + tokenizer.eos_token
    tokenized = tokenizer(full_text, truncation=True, max_length=max_length, return_tensors=None)
    input_ids = tokenized["input_ids"]

    # Find where answer starts
    prompt_tokens = tokenizer(prompt, truncation=True, max_length=max_length, return_tensors=None)["input_ids"]
    answer_start_pos = len(prompt_tokens)

    # Create labels: -100 for prompt, actual tokens for answer
    labels = [-100] * answer_start_pos + input_ids[answer_start_pos:]

    # Ensure labels has same length as input_ids
    labels = labels[:len(input_ids)]

    ex = {
        "input_ids": input_ids,
        "labels": labels,
    }

    return ex

######################################
# 3) Collator
######################################
@dataclass
class SimpleCollator:
    tokenizer: AutoTokenizer

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [x['input_ids'] for x in batch]
        labels = [x['labels'] for x in batch]

        input_ids, attn_mask = self.padding_input(input_ids)
        labels = self.padding_label(labels, max_len=input_ids.size(1))

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
        }

    def padding_input(self, sequences: List[List[int]]) -> (torch.Tensor, torch.Tensor):
        # return input, attention
        if len(sequences) == 0:
            return None, None
        max_len = max(len(s) for s in sequences)
        pad_id = self.tokenizer.pad_token_id
        batch_ids = []
        attn = []
        for s in sequences:
            pad_len = max_len - len(s)
            batch_ids.append(s + [pad_id] * pad_len)
            attn.append([1]*len(s) + [0]*pad_len)
        return torch.tensor(batch_ids, dtype=torch.long), torch.tensor(attn, dtype=torch.long)

    def padding_label(self, labels_list: List[List[int]], max_len: int = None) -> torch.Tensor:
        # return label
        if len(labels_list) == 0:
            return None
        if max_len is None:
            max_len = max(len(s) for s in labels_list)
        pad_val = -100
        out = []
        for s in labels_list:
            out.append(s + [pad_val] * (max_len - len(s)))
        return torch.tensor(out, dtype=torch.long)
    
######################################
# 4) Trainer
######################################
class DynamicTargetTrainer(Trainer):
    def __init__(self, idk_token_id, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.idk_token_id = idk_token_id
        self.tokenizer = tokenizer
        self.step_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute loss with dynamic target distribution.

        For each token position:
        - alpha = 0.5 * (1 - p_correct/p_top)
        - target[correct_token] = 1 - alpha
        - target[idk_token] = alpha
        """
        # Extract inputs
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        # Forward pass to get logits
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)

        batch_size, seq_len, vocab_size = logits.shape

        # Shift logits and labels for next-token prediction
        # logits[t] predicts token at position t+1
        shift_logits = logits[:, :-1, :].contiguous()  # (batch_size, seq_len-1, vocab_size)
        shift_labels = labels[:, 1:].contiguous()  # (batch_size, seq_len-1)

        # Update dimensions after shift
        seq_len = seq_len - 1

        # Compute probabilities
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        probs = torch.exp(log_probs)  # (batch_size, seq_len, vocab_size)

        # Create mask for valid positions (labels != -100)
        mask = (shift_labels != -100)  # (batch_size, seq_len)

        # Use shifted labels from here
        labels = shift_labels

        # Initialize target distribution and regularization indicator
        target_dist = torch.zeros_like(probs)  # (batch_size, seq_len, vocab_size)
        reg_indicator = torch.zeros(batch_size, seq_len, device=probs.device)  # (batch_size, seq_len)

        # For each valid position, compute dynamic target
        for b in range(batch_size):
            for t in range(seq_len):
                if not mask[b, t]:
                    continue

                label_token = labels[b, t].item()

                # Get probabilities for this position
                pos_probs = probs[b, t]  # (vocab_size,)

                # Get p_correct and p_top
                p_correct = pos_probs[label_token].item()
                p_top = pos_probs.max().item()

                # Compute alpha
                alpha = 0.5 * (1.0 - p_correct / p_top)

                # Clamp alpha to [0, 1]
                alpha = max(0.0, min(1.0, alpha))

                # Set target distribution
                target_dist[b, t, label_token] = 1.0 - alpha
                target_dist[b, t, self.idk_token_id] = alpha

                # Regularization indicator: I_correct = 1 if p_correct == p_top
                # Get probabilities for this position
                pos_probs = probs[b, t]  # (vocab_size,)

                # Get top predicted token
                top_token_idx = pos_probs.argmax().item()

                # Check if correct prediction
                if label_token == top_token_idx:
                    reg_indicator[b, t] = 1.0

        # Detailed logging every 10 steps
        self.step_count += 1
        if self.step_count % 10 == 1:
            # Collect statistics
            alpha_values = []
            p_correct_values = []
            p_top_values = []
            target_label_probs = []
            target_idk_probs = []

            for b in range(batch_size):
                for t in range(seq_len):
                    if mask[b, t]:
                        label_token = labels[b, t].item()
                        pos_probs = probs[b, t]
                        p_correct = pos_probs[label_token].item()
                        p_top = pos_probs.max().item()
                        alpha = 0.5 * (1.0 - p_correct / p_top)
                        alpha = max(0.0, min(1.0, alpha))

                        alpha_values.append(alpha)
                        p_correct_values.append(p_correct)
                        p_top_values.append(p_top)
                        target_label_probs.append(target_dist[b, t, label_token].item())
                        target_idk_probs.append(target_dist[b, t, self.idk_token_id].item())

            if alpha_values:
                # Calculate statistics
                alpha_mean = np.mean(alpha_values)
                alpha_min = np.min(alpha_values)
                alpha_max = np.max(alpha_values)
                p_correct_mean = np.mean(p_correct_values)
                p_top_mean = np.mean(p_top_values)
                target_label_mean = np.mean(target_label_probs)
                target_idk_mean = np.mean(target_idk_probs)
                correct_pred_ratio = reg_indicator[mask].mean().item()

                print(f"\n[Step {self.step_count}] Debug Info:")
                print(f"  Alpha: mean={alpha_mean:.4f}, min={alpha_min:.4f}, max={alpha_max:.4f}")
                print(f"  p_correct: mean={p_correct_mean:.6f}")
                print(f"  p_top: mean={p_top_mean:.6f}")
                print(f"  Target dist - label: {target_label_mean:.4f}, idk: {target_idk_mean:.4f}")
                print(f"  Correct prediction ratio: {correct_pred_ratio:.4f}")
                print(f"  Num masked tokens: {mask.sum().item()}")

                # Show sample predictions for first example
                if batch_size > 0:
                    first_mask = mask[0]
                    if first_mask.any():
                        # Get positions where mask is True
                        masked_positions = first_mask.nonzero(as_tuple=True)[0]
                        if len(masked_positions) > 0:
                            # Take first few positions
                            sample_pos = masked_positions[:5]
                            print(f"  Sample predictions (first {len(sample_pos)} tokens):")
                            for pos in sample_pos:
                                label_tok = labels[0, pos].item()
                                pred_tok = probs[0, pos].argmax().item()
                                label_text = self.tokenizer.decode([label_tok])
                                pred_text = self.tokenizer.decode([pred_tok])
                                p_label = probs[0, pos, label_tok].item()
                                p_pred = probs[0, pos, pred_tok].item()
                                p_idk = probs[0, pos, self.idk_token_id].item()
                                print(f"    pos {pos.item()}: label='{label_text}'({p_label:.4f}) pred='{pred_text}'({p_pred:.4f}) p_idk={p_idk:.6f}")

        # Compute cross entropy loss
        # CE = -sum(target * log_prob)
        # Add epsilon to IDK token log_probs to prevent -inf when p_idk ≈ 0
        log_probs_safe = log_probs.clone()
        p_idk_for_ce = probs[:, :, self.idk_token_id]
        log_probs_safe[:, :, self.idk_token_id] = torch.log(p_idk_for_ce) #  + 1e-3
        ce_loss = -(target_dist * log_probs_safe).sum(dim=-1)  # (batch_size, seq_len)

        # Compute regularization loss
        # L_reg = -I_correct * log(1 - p_idk)
        p_idk = probs[:, :, self.idk_token_id]  # (batch_size, seq_len)
        reg_loss = -reg_indicator * torch.log(1.0 - p_idk + 1e-10)  # (batch_size, seq_len)

        # Apply mask
        masked_ce_loss = ce_loss * mask.float()  # (batch_size, seq_len)
        masked_reg_loss = reg_loss * mask.float()  # (batch_size, seq_len)

        # Compute per-sample loss (mean over sequence)
        num_tokens = mask.float().sum(dim=1)  # (batch_size,)
        per_sample_ce_loss = masked_ce_loss.sum(dim=1) / (num_tokens + 1e-8)  # (batch_size,)
        per_sample_reg_loss = masked_reg_loss.sum(dim=1) / (num_tokens + 1e-8)  # (batch_size,)

        # Total loss (CE + Reg)
        total_loss = (per_sample_ce_loss + per_sample_reg_loss).mean()

        # Logging
        logs = {
            "loss": total_loss.detach().item(),
            "ce_loss": per_sample_ce_loss.mean().detach().item(),
            "reg_loss": per_sample_reg_loss.mean().detach().item(),
        }
        self.log(logs)

        return (total_loss, outputs) if return_outputs else total_loss



def main(args):
    # Initialize wandb
    wandb_run_name = f"{args.dataname}_{args.save_run_name}-ep{args.epochs}"
    wandb.init(
        project=args.wandb_project,
        name=wandb_run_name,
        config={
            "model_path": args.model_path,
            "train_data_path": args.train_data_path,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "grad_accum": args.grad_accum,
            "max_length": args.max_length,
        }
    )

    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set: tokenizer padding")

    # Add <IDK> special token
    special_token = "<IDK>"
    tokenizer.add_special_tokens({"additional_special_tokens": [special_token]})
    print(f"Added <IDK> special token with ID: {tokenizer.convert_tokens_to_ids(special_token)}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
    )

    # Resize model embeddings to accommodate new token
    model.resize_token_embeddings(len(tokenizer))

    # Get the token ID for <IDK> (needed for trainer)
    idk_token_id = tokenizer.convert_tokens_to_ids(special_token)
    print(f"<IDK> token ID: {idk_token_id}")

    # Unlike KWT, the <IDK> embedding is not initialized from "I don't know" here.

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    # Make sure model is in training mode
    model.train()

    train_data_path = args.train_data_path
    print(f"Loading training data from {train_data_path}")
    train_data = load_data(train_data_path)
    print(f"Loaded {len(train_data)} training examples")

    train_tokenized = []
    for ex in train_data:
        train_tokenized.append(preprocess_tokenize(ex, tokenizer, max_length=args.max_length))
    train_ds = Dataset.from_list(train_tokenized)

    collator = SimpleCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=args.logging_steps,
        save_strategy="no",
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to="wandb",
    )

    trainer = DynamicTargetTrainer(
        idk_token_id=idk_token_id,
        tokenizer=tokenizer,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SEAL baseline: token-level <IDK> probability reallocation')
    parser.add_argument('--dataname', type=str, required=True, choices=['halueval', 'medqa', 'sciq'])
    parser.add_argument('--model_path', type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument('--ckpt_root', type=str, default=str(CKPT_DIR), help='Root directory for checkpoints')
    parser.add_argument('--max_length', type=int, default=1024)

    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--wandb_project', type=str, default="kwt", help='WandB project name')

    args = parser.parse_args()
    args.save_run_name = "seal"

    # SEAL only uses (question, answer) pairs; the knowledge file is read for convenience.
    model_name = get_model_name(args.model_path)
    args.train_data_path = str(knowledge_file(args.dataname, model_name, "llm"))
    args.output_dir = str(checkpoint_dir(args.ckpt_root, args.dataname, model_name, "seal"))

    print(f"Dataset: {args.dataname}")
    print(f"Train data path: {args.train_data_path}")
    print(f"Output directory: {args.output_dir}")

    main(args)
