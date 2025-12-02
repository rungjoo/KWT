#!/usr/bin/env python3
"""
Test script to debug DynamicTargetTrainer behavior.
This script runs a few training steps and logs detailed information about:
- alpha values distribution
- target distribution
- p_idk changes
- loss components
"""

import os
import torch
import json
import argparse
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from dataclasses import dataclass
from typing import Dict, List, Any
import warnings
warnings.filterwarnings("ignore")

# Import from train_seal.py
def load_data(filepath):
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

def create_prompt(question):
    return f"Question: {question}\n\nAnswer:"

def preprocess_tokenize(item, tokenizer, max_length=1024):
    question = item['question']
    prompt = create_prompt(question)

    full_text = prompt + " " + item['response'] + tokenizer.eos_token
    tokenized = tokenizer(full_text, truncation=True, max_length=max_length, return_tensors=None)
    input_ids = tokenized["input_ids"]

    prompt_tokens = tokenizer(prompt, truncation=True, max_length=max_length, return_tensors=None)["input_ids"]
    answer_start_pos = len(prompt_tokens)

    labels = [-100] * answer_start_pos + input_ids[answer_start_pos:]
    labels = labels[:len(input_ids)]

    return {
        "input_ids": input_ids,
        "labels": labels,
    }

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

    def padding_input(self, sequences):
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

    def padding_label(self, labels_list, max_len=None):
        if len(labels_list) == 0:
            return None
        if max_len is None:
            max_len = max(len(s) for s in labels_list)
        pad_val = -100
        out = []
        for s in labels_list:
            out.append(s + [pad_val] * (max_len - len(s)))
        return torch.tensor(out, dtype=torch.long)


class DebugDynamicTargetTrainer(Trainer):
    """
    Debug version of DynamicTargetTrainer with detailed logging.
    """
    def __init__(self, idk_token_id, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.idk_token_id = idk_token_id
        self.debug_tokenizer = tokenizer
        self.step_count = 0

        # Statistics tracking
        self.alpha_history = []
        self.p_idk_history = []
        self.loss_history = []

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        batch_size, seq_len, vocab_size = logits.shape

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        mask = (labels != -100)

        target_dist = torch.zeros_like(probs)
        reg_indicator = torch.zeros(batch_size, seq_len, device=probs.device)

        # For debugging
        alpha_values = []
        p_correct_values = []
        p_top_values = []
        p_idk_values = []
        correct_predictions = 0
        total_predictions = 0

        for b in range(batch_size):
            for t in range(seq_len):
                if not mask[b, t]:
                    continue

                label_token = labels[b, t].item()
                pos_probs = probs[b, t]

                p_correct = pos_probs[label_token].item()
                p_top = pos_probs.max().item()
                p_idk = pos_probs[self.idk_token_id].item()

                alpha = 0.5 * (1.0 - p_correct / p_top)
                alpha = max(0.0, min(1.0, alpha))

                target_dist[b, t, label_token] = 1.0 - alpha
                target_dist[b, t, self.idk_token_id] = alpha

                top_token_idx = pos_probs.argmax().item()
                if label_token == top_token_idx:
                    reg_indicator[b, t] = 1.0
                    correct_predictions += 1

                total_predictions += 1

                # Track statistics
                alpha_values.append(alpha)
                p_correct_values.append(p_correct)
                p_top_values.append(p_top)
                p_idk_values.append(p_idk)

        # Compute losses
        ce_loss = -(target_dist * log_probs).sum(dim=-1)
        p_idk = probs[:, :, self.idk_token_id]
        reg_loss = -reg_indicator * torch.log(1.0 - p_idk + 1e-10)

        masked_ce_loss = ce_loss * mask.float()
        masked_reg_loss = reg_loss * mask.float()

        num_tokens = mask.float().sum(dim=1)
        per_sample_ce_loss = masked_ce_loss.sum(dim=1) / (num_tokens + 1e-8)
        per_sample_reg_loss = masked_reg_loss.sum(dim=1) / (num_tokens + 1e-8)

        total_loss = (per_sample_ce_loss + per_sample_reg_loss).mean()

        # Log detailed statistics every step
        self.step_count += 1

        if alpha_values:
            avg_alpha = sum(alpha_values) / len(alpha_values)
            avg_p_correct = sum(p_correct_values) / len(p_correct_values)
            avg_p_top = sum(p_top_values) / len(p_top_values)
            avg_p_idk = sum(p_idk_values) / len(p_idk_values)
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

            # Count alpha distribution
            alpha_zero = sum(1 for a in alpha_values if a == 0)
            alpha_nonzero = len(alpha_values) - alpha_zero

            print(f"\n{'='*60}")
            print(f"Step {self.step_count} Debug Info:")
            print(f"{'='*60}")
            print(f"Total tokens: {total_predictions}")
            print(f"Correct predictions: {correct_predictions} ({accuracy:.2%})")
            print(f"\nAlpha distribution:")
            print(f"  - alpha = 0 (correct): {alpha_zero} ({alpha_zero/len(alpha_values):.2%})")
            print(f"  - alpha > 0 (wrong):   {alpha_nonzero} ({alpha_nonzero/len(alpha_values):.2%})")
            print(f"  - avg alpha: {avg_alpha:.4f}")
            print(f"  - max alpha: {max(alpha_values):.4f}")
            print(f"\nProbability stats:")
            print(f"  - avg p_correct: {avg_p_correct:.4f}")
            print(f"  - avg p_top:     {avg_p_top:.4f}")
            print(f"  - avg p_idk:     {avg_p_idk:.4f}")
            print(f"  - max p_idk:     {max(p_idk_values):.4f}")
            print(f"\nLoss components:")
            print(f"  - CE loss:  {per_sample_ce_loss.mean().item():.4f}")
            print(f"  - Reg loss: {per_sample_reg_loss.mean().item():.4f}")
            print(f"  - Total:    {total_loss.item():.4f}")

            # Sample some predictions
            print(f"\nSample predictions (first 5 tokens):")
            b = 0
            count = 0
            for t in range(seq_len):
                if mask[b, t] and count < 5:
                    label_id = labels[b, t].item()
                    pred_id = probs[b, t].argmax().item()
                    label_tok = self.debug_tokenizer.decode([label_id])
                    pred_tok = self.debug_tokenizer.decode([pred_id])
                    idk_prob = probs[b, t, self.idk_token_id].item()

                    status = "OK" if label_id == pred_id else "WRONG"
                    print(f"  [{count}] Label: '{label_tok}' ({label_id}), "
                          f"Pred: '{pred_tok}' ({pred_id}), "
                          f"p_idk: {idk_prob:.4f} [{status}]")
                    count += 1

            # Track history
            self.alpha_history.append(avg_alpha)
            self.p_idk_history.append(avg_p_idk)
            self.loss_history.append(total_loss.item())

        return (total_loss, outputs) if return_outputs else total_loss


def main():
    parser = argparse.ArgumentParser(description='Test DynamicTargetTrainer')
    parser.add_argument('--dataname', type=str, default="halueval")
    parser.add_argument('--model_path', type=str, default="../../model/Llama-3.2-3B")
    parser.add_argument('--max_steps', type=int, default=10, help='Number of steps to run')
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()

    print("="*60)
    print("DynamicTargetTrainer Debug Test")
    print("="*60)

    # Load tokenizer and model
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add <IDK> token
    special_token = "<IDK>"
    tokenizer.add_special_tokens({"additional_special_tokens": [special_token]})
    idk_token_id = tokenizer.convert_tokens_to_ids(special_token)
    print(f"<IDK> token ID: {idk_token_id}")

    # Also check what "I" token ID is
    i_token_id = tokenizer.encode("I", add_special_tokens=False)[0]
    print(f"'I' token ID: {i_token_id}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.resize_token_embeddings(len(tokenizer))

    # Initialize <IDK> embedding
    idk_text = "I don't know"
    idk_tokens = tokenizer(idk_text, return_tensors="pt", add_special_tokens=False)
    print(f"'I don't know' token IDs: {idk_tokens['input_ids'][0].tolist()}")

    with torch.no_grad():
        input_embeddings = model.get_input_embeddings()
        idk_embeddings = input_embeddings(idk_tokens['input_ids'][0].to(model.device))
        mean_embedding = idk_embeddings.mean(dim=0)
        input_embeddings.weight[idk_token_id] = mean_embedding

    print("Initialized <IDK> token embedding")

    # Load data
    train_data_path = f"../dataset_type/{args.dataname}/base_model_temp0.7_samples5_fewshot3_evaluated_llm.json"
    print(f"\nLoading data from {train_data_path}")
    train_data = load_data(train_data_path)

    # Use only a few samples for debugging
    train_data = train_data[:20]
    print(f"Using {len(train_data)} samples for testing")

    # Tokenize
    train_tokenized = []
    for ex in train_data:
        train_tokenized.append(preprocess_tokenize(ex, tokenizer, max_length=512))
    train_ds = Dataset.from_list(train_tokenized)

    collator = SimpleCollator(tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./debug_output",
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        remove_unused_columns=False,
        report_to="none",  # Disable wandb
    )

    # Create trainer
    trainer = DebugDynamicTargetTrainer(
        idk_token_id=idk_token_id,
        tokenizer=tokenizer,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
    )

    print("\n" + "="*60)
    print("Starting training debug...")
    print("="*60)

    trainer.train()

    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)

    if trainer.alpha_history:
        print(f"\nAlpha trend: {trainer.alpha_history[0]:.4f} -> {trainer.alpha_history[-1]:.4f}")
        print(f"p_idk trend: {trainer.p_idk_history[0]:.4f} -> {trainer.p_idk_history[-1]:.4f}")
        print(f"Loss trend:  {trainer.loss_history[0]:.4f} -> {trainer.loss_history[-1]:.4f}")

        # Check if p_idk is increasing
        if len(trainer.p_idk_history) > 1:
            if trainer.p_idk_history[-1] > trainer.p_idk_history[0] * 1.5:
                print("\nWARNING: p_idk is increasing significantly!")
                print("This indicates the model is collapsing towards <IDK> token.")

    print("\nDebug test completed.")


if __name__ == "__main__":
    main()
