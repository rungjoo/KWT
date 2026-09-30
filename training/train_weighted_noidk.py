import sys
import torch
import json
import argparse
import wandb
from pathlib import Path
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import CKPT_DIR, get_model_name, resolve_threshold, knowledge_file, checkpoint_dir

######################################
# 1) Data loading
######################################

def calculate_sample_weight_reverse_smooth(samples_correct, samples_total, sft_idk_weight):
    # ranges from 1/6 (0/5 correct) to 6/6 (5/5 correct)
    sample_weight = (samples_correct+1) / (samples_total+1)
    return sample_weight

def calculate_sample_weight_smooth(samples_correct, samples_total, sft_idk_weight):
    # ranges from 1 (0/5 correct) to 1/6 (5/5 correct)
    sample_weight = 1 - (samples_correct) / (samples_total+1)
    return sample_weight

def calculate_sample_weight_uniform(samples_correct, samples_total, sft_idk_weight):
    if samples_correct > 0:
        sample_weight = 1.0
    else:
        sample_weight = sft_idk_weight
    return sample_weight

# Dictionary mapping strategy names to functions
SAMPLE_WEIGHT_STRATEGIES = {
    'sample_weight_reverse_smooth': calculate_sample_weight_reverse_smooth,
    'sample_weight_smooth': calculate_sample_weight_smooth,
    'sample_uniform': calculate_sample_weight_uniform,
}


def load_data(filepath, sft_idk_weight=0.1, weight_strategy='sample_weight_reverse_smooth'):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Get the weight calculation function
    if weight_strategy not in SAMPLE_WEIGHT_STRATEGIES:
        raise ValueError(f"Unknown weight_strategy: {weight_strategy}. Available: {list(SAMPLE_WEIGHT_STRATEGIES.keys())}")

    weight_fn = SAMPLE_WEIGHT_STRATEGIES[weight_strategy]

    data_type = []
    for result in data['results']:
        question = result['question']
        right_answer = result['right_answer']

        # Get samples_correct (number of correct samples out of 5)
        samples_correct = result.get('samples_correct', 0)
        samples_total = result.get('samples_total', 5)

        # Calculate sample weight using the selected strategy
        sample_weight = weight_fn(samples_correct, samples_total, sft_idk_weight)

        # Skip data with zero or negative weight
        if sample_weight <= 0:
            continue

        # No <IDK> - always use right_answer as response
        item = {
            'question': question,
            'response': right_answer,
            'sample_weight': sample_weight,
            'samples_correct': samples_correct,
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

    # Get sample weight
    sample_weight = item.get('sample_weight', 1.0)

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
        "sample_weight": sample_weight,
    }

    return ex

######################################
# 3) Collator
######################################
@dataclass
class SampleWeightedCollator:
    tokenizer: AutoTokenizer

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [x['input_ids'] for x in batch]
        labels = [x['labels'] for x in batch]
        weights = torch.tensor([x['sample_weight'] for x in batch], dtype=torch.float32)

        padded_input, attn = self.padding_input(input_ids)
        padded_labels = self.padding_label(labels, max_len=padded_input.size(1))

        return {
            "input_ids": padded_input,
            "attention_mask": attn,
            "labels": padded_labels,
            "sample_weights": weights,
        }

    def padding_input(self, sequences: List[List[int]]) -> (torch.Tensor, torch.Tensor):
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
class SampleWeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sample_weights = inputs.pop("sample_weights")
        device = next(model.parameters()).device
        sample_weights = sample_weights.to(device)

        outputs = model(**inputs)
        loss = outputs.loss * sample_weights.mean()

        logs = {
            "sft/loss": loss.detach().item(),
            "sft/avg_weight": sample_weights.mean().item(),
        }
        self.log(logs)

        return (loss, None) if return_outputs else loss



def main(args):
    # Initialize wandb
    wandb_run_name = f"{args.dataname}_{args.save_run_name}_noidk-ep{args.epochs}_idk{args.sft_idk_weight}"
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
            "sft_idk_weight": args.sft_idk_weight,
            "weight_strategy": args.save_run_name,
            "max_length": args.max_length,
        }
    )

    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set: tokenizer padding")

    # No <IDK> special token - just use right_answer as response

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
    )

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    # Make sure model is in training mode
    model.train()

    train_data_path = args.train_data_path
    print(f"Loading training data from {train_data_path}")
    print(f"Using weight strategy: {args.save_run_name}")
    train_data = load_data(train_data_path, sft_idk_weight=args.sft_idk_weight, weight_strategy=args.save_run_name)
    print(f"Loaded {len(train_data)} training examples")

    train_tokenized = []
    for ex in train_data:
        train_tokenized.append(preprocess_tokenize(ex, tokenizer, max_length=args.max_length))
    train_ds = Dataset.from_list(train_tokenized)

    collator = SampleWeightedCollator(tokenizer)

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

    trainer = SampleWeightedTrainer(
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
    parser = argparse.ArgumentParser(description='KWT (non-IDK): knowledge-weighted SFT without <IDK> supervision')
    parser.add_argument('--dataname', type=str, required=True, choices=['halueval', 'medqa', 'sciq'])
    parser.add_argument('--model_path', type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument('--save_run_name', type=str, default="sample_weight_reverse_smooth",
                        choices=list(SAMPLE_WEIGHT_STRATEGIES.keys()),
                        help='Weight strategy: sample_weight_reverse_smooth (F) / sample_weight_smooth (RF) / sample_uniform (U)')
    parser.add_argument('--eval_method', type=str, default="llm", choices=['llm', 'rouge', 'em'])
    parser.add_argument('--threshold', type=str, default=None,
                        help='ROUGE-L threshold (default: 0.35 for halueval, 0.6 for medqa/sciq)')
    parser.add_argument('--sft_idk_weight', type=float, default=0.16,
                        help='Weight of KS=0 samples for sample_uniform (kept in the output name)')
    parser.add_argument('--ckpt_root', type=str, default=str(CKPT_DIR), help='Root directory for checkpoints')

    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--wandb_project', type=str, default="kwt", help='WandB project name')

    args = parser.parse_args()

    model_name = get_model_name(args.model_path)
    args.threshold = resolve_threshold(args.dataname, args.eval_method, args.threshold)
    args.train_data_path = str(knowledge_file(args.dataname, model_name, args.eval_method, args.threshold))
    # saved as "<save_run_name>_noidk" (evaluate with --save_run_name <save_run_name>_noidk)
    args.output_dir = str(checkpoint_dir(args.ckpt_root, args.dataname, model_name, f"{args.save_run_name}_noidk",
                                         args.eval_method, args.threshold, args.sft_idk_weight))

    print(f"Dataset: {args.dataname}")
    print(f"Train data path: {args.train_data_path}")
    print(f"Output directory: {args.output_dir}")

    main(args)
