import os
import torch
import json
import math
import copy, random
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

def get_model_name(model_path):
    """Extract model name from model path for directory naming"""
    return Path(model_path).name.lower()

######################################
# 1) 데이터 로딩
######################################

def calculate_sample_weight_noidk(samples_correct, samples_total, sft_idk_weight):
    sample_weight = 1.0
    loss_type = 'sft'
    should_add_idk = False

    return sample_weight, loss_type, should_add_idk

def calculate_sample_weight_default(samples_correct, samples_total, sft_idk_weight):
    """
    Default sample weighting strategy.
    Higher weight for harder questions (fewer correct samples).

    Args:
        samples_correct: Number of correct samples out of total
        samples_total: Total number of samples
        sft_idk_weight: Weight to use for IDK cases (samples_correct = 0)

    Returns:
        tuple: (sample_weight, loss_type, should_add_idk_token)
            - sample_weight: Weight value for this sample
            - loss_type: 'sft' or 'sft_idk'
            - should_add_idk_token: Whether to append <IDK> to the response
    """
    if samples_correct > 0:
        accuracy = samples_correct / samples_total
        # ranges from 1.0 (1/5 correct) to 0.2 (5/5 correct)
        sample_weight = 1.2 - accuracy
        loss_type = 'sft'
        should_add_idk = False
    else:
        sample_weight = sft_idk_weight
        loss_type = "sft_idk"
        should_add_idk = True

    return sample_weight, loss_type, should_add_idk

def calculate_sample_weight_reverse(samples_correct, samples_total, sft_idk_weight):
    if samples_correct > 0:
        accuracy = samples_correct / samples_total
        # ranges from 1.0 (5/5 correct) to 0.2 (1/5 correct)
        sample_weight = accuracy
        loss_type = 'sft'
        should_add_idk = False
    else:
        sample_weight = sft_idk_weight
        loss_type = "sft_idk"
        should_add_idk = True

    return sample_weight, loss_type, should_add_idk

def calculate_sample_weight_popular(samples_correct, samples_total, sft_idk_weight):
    if samples_correct > 0:
        sample_weight = 1.0
        loss_type = 'sft'
        should_add_idk = False
    else:
        return 0, 0, 0

    return sample_weight, loss_type, should_add_idk

def calculate_sample_weight_uniform(samples_correct, samples_total, sft_idk_weight):
    if samples_correct > 0:
        sample_weight = 1.0
        loss_type = 'sft'
        should_add_idk = False
    else:
        sample_weight = sft_idk_weight
        loss_type = "sft_idk"
        should_add_idk = True

    return sample_weight, loss_type, should_add_idk

# Dictionary mapping strategy names to functions
SAMPLE_WEIGHT_STRATEGIES = {
    'sft': calculate_sample_weight_noidk,
    'sample_weighted': calculate_sample_weight_default,
    'sample_uniform': calculate_sample_weight_uniform,
    'rtuning_r': calculate_sample_weight_uniform,
    'sample_weighted_reverse': calculate_sample_weight_reverse,
    'sample_weighted_reverse_noinit': calculate_sample_weight_reverse,
    'popular': calculate_sample_weight_popular
}


def load_data(filepath, sft_idk_weight=0.1, weight_strategy='default'):
    """
    Load and prepare training data with configurable sample weighting.

    Args:
        filepath: Path to the JSON file containing training data
        sft_idk_weight: Weight for IDK cases (samples_correct = 0)
        weight_strategy: Strategy for calculating sample weights. Options:
            - 'default': Higher weight for harder questions (1.2 - accuracy)
            - 'uniform': Equal weight for all samples (1.0)

    Returns:
        list: List of processed data items with weights
    """
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
        sample_weight, loss_type, should_add_idk = weight_fn(
            samples_correct, samples_total, sft_idk_weight
        )

        # Skip data with zero or negative weight (excluded from training)
        if sample_weight <= 0:
            continue

        # Prepare response with or without <IDK> token
        response = f"{right_answer} <IDK>" if should_add_idk else right_answer

        item = {
            'question': question,
            'response': response,
            'loss_type': loss_type,
            'sample_weight': sample_weight,
            'samples_correct': samples_correct,
        }

        data_type.append(item)

    return data_type

######################################
# 2) 프롬프트/토크나이즈
######################################
def create_prompt(question):
    return f"Question: {question}\n\nAnswer:"

def preprocess_tokenize(item, tokenizer, max_length=1024):
    loss_type = item['loss_type']
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
        "loss_type": loss_type,
        "input_ids": input_ids,
        "labels": labels,
        "sample_weight": sample_weight,
    }

    return ex

######################################
# 3) Collator: 배치 패딩 처리
######################################
@dataclass
class SampleWeightedCollator:
    tokenizer: AutoTokenizer

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        sft_items = [b for b in batch if b["loss_type"] == "sft"]
        sft_idk_items = [b for b in batch if b["loss_type"] == "sft_idk"]

        collated: Dict[str, Any] = {}

        if len(sft_items) > 0:
            sft_input_ids = [x['input_ids'] for x in sft_items]
            sft_labels = [x['labels'] for x in sft_items]
            sft_weights = torch.tensor([x['sample_weight'] for x in sft_items], dtype=torch.float32)
            sft_input, sft_attn = self.padding_input(sft_input_ids)
            sft_labels = self.padding_label(sft_labels, max_len=sft_input.size(1))
            collated["sft"] = {
                "input_ids": sft_input,
                "attention_mask": sft_attn,
                "labels": sft_labels,
                "sample_weights": sft_weights,
            }

        if len(sft_idk_items) > 0:
            sft_idk_input_ids = [x['input_ids'] for x in sft_idk_items]
            sft_idk_labels = [x['labels'] for x in sft_idk_items]
            sft_idk_weights = torch.tensor([x['sample_weight'] for x in sft_idk_items], dtype=torch.float32)
            sft_idk_input, sft_idk_attn = self.padding_input(sft_idk_input_ids)
            sft_idk_labels = self.padding_label(sft_idk_labels, max_len=sft_idk_input.size(1))
            collated["sft_idk"] = {
                "input_ids": sft_idk_input,
                "attention_mask": sft_idk_attn,
                "labels": sft_idk_labels,
                "sample_weights": sft_idk_weights,
            }

        return collated

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
# 4) 커스텀 Trainer
######################################
class SampleWeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        total_loss = 0.0
        logs = {}
        device = next(model.parameters()).device

        # SFT loss with sample weights
        if "sft" in inputs:
            sft_batch = {k: v for k, v in inputs["sft"].items() if k != "sample_weights"}
            sft_batch = self._prepare_inputs(sft_batch)
            sample_weights = inputs["sft"]["sample_weights"].to(device)
            outputs = model(**sft_batch)
            # Apply per-sample weights (already contains difficulty-based weighting)
            sft_loss = outputs.loss * sample_weights.mean()
            total_loss = total_loss + sft_loss
            logs["sft/loss"] = sft_loss.detach().item()
            logs["sft/avg_weight"] = sample_weights.mean().item()

        # SFT_IDK loss with sample weights
        if "sft_idk" in inputs:
            sft_idk_batch = {k: v for k, v in inputs["sft_idk"].items() if k != "sample_weights"}
            sft_idk_batch = self._prepare_inputs(sft_idk_batch)
            sample_weights = inputs["sft_idk"]["sample_weights"].to(device)
            outputs = model(**sft_idk_batch)
            # Apply per-sample weights (already contains sft_idk_weight)
            sft_idk_loss = outputs.loss * sample_weights.mean()
            total_loss = total_loss + sft_idk_loss
            logs["sft_idk/loss"] = sft_idk_loss.detach().item()
            logs["sft_idk/avg_weight"] = sample_weights.mean().item()

        self.log(logs)
        return (total_loss, None) if return_outputs else total_loss



def main(args):
    # Determine weight strategy from save_run_name
    # Initialize wandb
    wandb_run_name = f"{args.dataname}_{args.save_run_name}-ep{args.epochs}_idk{args.sft_idk_weight}"
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
        tokenizer.pad_token = tokenizer.eos_token  # 안전하게 eos를 pad로 사용
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

    # sample_weighted_reverse_noinit 세팅에선 초기화 안함
    if args.save_run_name != "sample_weighted_reverse_noinit":
        # Initialize <IDK> token embedding with average of "I don't know"
        idk_text = "I don't know"
        idk_tokens = tokenizer(idk_text, return_tensors="pt", add_special_tokens=False)

        with torch.no_grad():
            # Get input embeddings
            input_embeddings = model.get_input_embeddings()

            # Get embeddings for "I don't know" tokens
            idk_embeddings = input_embeddings(idk_tokens['input_ids'][0].to(model.device))

            # Calculate mean embedding
            mean_embedding = idk_embeddings.mean(dim=0)

            # Get the token ID for <IDK>
            idk_token_id = tokenizer.convert_tokens_to_ids(special_token)

            # Initialize <IDK> token embedding with the mean
            input_embeddings.weight[idk_token_id] = mean_embedding

        print("Initialized <IDK> token embedding with average of 'I don't know'")

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
        logging_steps= args.logging_steps,
        # save_steps= args.save_steps,
        save_strategy="no", 
        save_total_limit=2,
        bf16=torch.cuda.is_available(),  # A100/H100면 bf16, 아니면 자동으로 fp32 사용
        remove_unused_columns=False,     # collator가 dict 구조를 유지하도록
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
    # python3 train_sample.py --dataname halueval --epochs 3 --sft_idk_weight 0.1 --save_run_name sample_weighted --eval_method rouge --threshold 0.6
    # python3 train_sample.py --dataname medqa --epochs 3 --sft_idk_weight 0.1 --save_run_name sample_weighted --eval_method rouge --threshold 0.6
    # python3 train_sample.py --dataname sciq --epochs 3 --sft_idk_weight 0.1 --save_run_name sample_weighted --eval_method rouge --threshold 0.6

    # python3 train_sample.py --dataname halueval --epochs 3 --sft_idk_weight 0.1 --save_run_name rtuning_r --eval_method rouge --threshold 0.6
    # python3 train_sample.py --dataname medqa --epochs 3 --sft_idk_weight 0.1 --save_run_name rtuning_r --eval_method rouge --threshold 0.6
    # python3 train_sample.py --dataname sciq --epochs 3 --sft_idk_weight 0.1 --save_run_name rtuning_r --eval_method rouge --threshold 0.6
    parser = argparse.ArgumentParser(description='Train model with sample-weighted SFT')
    parser.add_argument('--dataname', type=str, required=True, help='Dataset name (e.g., halueval, medqa)')
    parser.add_argument('--eval_method', type=str, default="rouge")
    parser.add_argument('--model_path', type=str, default="../../model/Llama-3.2-3B")
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--threshold', type=str, default="None",
                       help='Threshold for bertscore/rouge (default: 0.7 for bertscore, 0.6 for rouge)')

    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)

    parser.add_argument('--logging_steps', type=int, default=10)

    # Sample weight for IDK cases
    parser.add_argument('--sft_idk_weight', type=float, default=0.1, help='Weight for samples with no correct answers (IDK cases)')

    # Wandb configuration
    parser.add_argument('--wandb_project', type=str, default="hall-data-cur", help='WandB project name')
    parser.add_argument('--save_run_name', type=str, default="sample_weighted", help='WandB run name (optional)') # sample_weighted / sample_uniform / rtuning_r

    args = parser.parse_args()

    # Set train_data_path and output_dir based on dataname    
    if args.eval_method in ["llm", "em"]:
        args.threshold = ""
    else:
        args.threshold = float(args.threshold)

    model_name = get_model_name(args.model_path)
    if args.save_run_name == "rtuning_r":
        args.train_data_path = f"../dataset_type/{args.dataname}/{model_name}/base_model_greedy_samples_fewshot3_evaluated_em.json"
        args.output_dir = f"/mnt/frdata/rungjoo/hall/halu_model/{args.dataname}/{model_name}/{args.save_run_name}/sw_{args.eval_method}{args.threshold}_idk{args.sft_idk_weight}"
    elif args.save_run_name == "sft":
        args.train_data_path = f"../dataset_type/{args.dataname}/{model_name}/base_model_temp0.7_samples5_fewshot3_evaluated_llm.json"
        args.output_dir = f"/mnt/frdata/rungjoo/hall/halu_model/{args.dataname}/{model_name}/{args.save_run_name}/sft"
    else:
        args.train_data_path = f"../dataset_type/{args.dataname}/{model_name}/base_model_temp0.7_samples5_fewshot3_evaluated_{args.eval_method}{args.threshold}.json"
        args.output_dir = f"/mnt/frdata/rungjoo/hall/halu_model/{args.dataname}/{model_name}/{args.save_run_name}/sw_{args.eval_method}{args.threshold}_idk{args.sft_idk_weight}"

    print(f"Dataset: {args.dataname}")
    print(f"Train data path: {args.train_data_path}")
    print(f"Output directory: {args.output_dir}")

    main(args)