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
# 1) Data loading / sample weighting
######################################
# Each strategy maps (samples_correct, samples_total, sft_idk_weight) to
# (sample_weight, loss_type, idk_mode), where idk_mode is one of
#   False      : response = answer
#   True       : response = "answer <IDK>"   (append-IDK, default KWT)
#   "ridk"     : response = "<IDK> answer"   (prepend-IDK)
#   "idk_only" : response = "<IDK>"          (only-IDK)
# samples_total is S (=5) and samples_correct / samples_total is the knowledge score KS.

def weight_sft(samples_correct, samples_total, sft_idk_weight):
    """SFT: uniform weight, no <IDK>."""
    return 1.0, "sft", False

def weight_popular(samples_correct, samples_total, sft_idk_weight):
    """FT-TOP: train only on samples the base model already knows."""
    if samples_correct > 0:
        return 1.0, "sft", False
    return 0, 0, 0

def weight_uniform(samples_correct, samples_total, sft_idk_weight):
    """KWT-U / R-Tuning: weight 1 for known samples, sft_idk_weight for unknown (+<IDK>)."""
    if samples_correct > 0:
        return 1.0, "sft", False
    return sft_idk_weight, "sft_idk", True

def familiarity_weight(samples_correct, samples_total):
    """Familiarity (F) weight, Eq. (3): (S*KS + 1) / (S + 1), ranges from 1/6 to 1."""
    return (samples_correct + 1) / (samples_total + 1)

def weight_familiarity(samples_correct, samples_total, sft_idk_weight):
    """KWT (F weighting, append-IDK)."""
    w = familiarity_weight(samples_correct, samples_total)
    if samples_correct > 0:
        return w, "sft", False
    return w, "sft_idk", True

def weight_reverse_familiarity(samples_correct, samples_total, sft_idk_weight):
    """KWT-RF, Eq. (4): 1 - S*KS / (S + 1), ranges from 1 to 1/6."""
    w = 1 - samples_correct / (samples_total + 1)
    if samples_correct > 0:
        return w, "sft", False
    return w, "sft_idk", True

def weight_familiarity_prepend_idk(samples_correct, samples_total, sft_idk_weight):
    """KWT (prepend-IDK), Appendix D."""
    w = familiarity_weight(samples_correct, samples_total)
    if samples_correct > 0:
        return w, "sft", False
    return w, "sft_idk", "ridk"

def weight_familiarity_only_idk(samples_correct, samples_total, sft_idk_weight):
    """KWT (only-IDK), Appendix D."""
    w = familiarity_weight(samples_correct, samples_total)
    if samples_correct > 0:
        return w, "sft", False
    return w, "sft_idk", "idk_only"

# --save_run_name -> strategy. Names are kept as-is so that checkpoint / result
# file names stay consistent with the released experiments.
SAMPLE_WEIGHT_STRATEGIES = {
    'sft': weight_sft,                                              # SFT
    'popular': weight_popular,                                      # FT-TOP
    'rtuning_r': weight_uniform,                                    # R-Tuning (greedy EM knowledge)
    'sample_uniform': weight_uniform,                               # KWT-U
    'sample_weight_reverse_smooth': weight_familiarity,             # KWT (F, default)
    'sample_weight_smooth': weight_reverse_familiarity,             # KWT-RF
    'sample_weighted_reverse_ridk': weight_familiarity_prepend_idk, # KWT (prepend-IDK)
    'sample_weighted_reverse_idkonly': weight_familiarity_only_idk, # KWT (only-IDK)
}


def load_data(filepath, sft_idk_weight=0.1, weight_strategy='sample_weight_reverse_smooth'):
    """Load a judged knowledge-estimation file and attach per-sample weights."""
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
        if should_add_idk == "idk_only":
            response = "<IDK>"
        elif should_add_idk == "ridk":
            response = f"<IDK> {right_answer}"
        elif should_add_idk:
            response = f"{right_answer} <IDK>"
        else:
            response = right_answer

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
# 2) Prompt / tokenization
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
# 3) Collator
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
# 4) Trainer
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

    # Initialize <IDK> token embedding with the mean embedding of "I don't know"
    idk_tokens = tokenizer("I don't know", return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        input_embeddings = model.get_input_embeddings()
        idk_embeddings = input_embeddings(idk_tokens['input_ids'][0].to(model.device))
        idk_token_id = tokenizer.convert_tokens_to_ids(special_token)
        input_embeddings.weight[idk_token_id] = idk_embeddings.mean(dim=0)
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
        logging_steps=args.logging_steps,
        save_strategy="no",
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,     # keep the nested dict produced by the collator
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
    parser = argparse.ArgumentParser(description='Knowledge-weighted fine-tuning (KWT) and baselines')
    parser.add_argument('--dataname', type=str, required=True, choices=['halueval', 'medqa', 'sciq'])
    parser.add_argument('--model_path', type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument('--save_run_name', type=str, default="sample_weight_reverse_smooth",
                        choices=list(SAMPLE_WEIGHT_STRATEGIES.keys()),
                        help='Training strategy (see SAMPLE_WEIGHT_STRATEGIES)')
    parser.add_argument('--eval_method', type=str, default="llm", choices=['llm', 'rouge', 'em'],
                        help='Matching function used to compute the knowledge score')
    parser.add_argument('--threshold', type=str, default=None,
                        help='ROUGE-L threshold (default: 0.35 for halueval, 0.6 for medqa/sciq)')
    parser.add_argument('--sft_idk_weight', type=float, default=0.16,
                        help='Weight of unknown (KS=0) samples for sample_uniform / rtuning_r. '
                             'Other strategies ignore it, but it is kept in the output name.')
    parser.add_argument('--run_id', type=int, default=0, help='Run ID for repeated SFT runs (0 means no suffix)')
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
    if args.save_run_name == "rtuning_r":
        # R-Tuning: binary known/unknown from a single greedy response, judged by EM
        args.eval_method = "em"
        args.threshold = ""
        args.train_data_path = str(knowledge_file(args.dataname, model_name, "em", greedy=True))
    else:
        args.threshold = resolve_threshold(args.dataname, args.eval_method, args.threshold)
        args.train_data_path = str(knowledge_file(args.dataname, model_name, args.eval_method, args.threshold))
    args.output_dir = str(checkpoint_dir(args.ckpt_root, args.dataname, model_name, args.save_run_name,
                                         args.eval_method, args.threshold, args.sft_idk_weight, args.run_id))

    print(f"Dataset: {args.dataname}")
    print(f"Train data path: {args.train_data_path}")
    print(f"Output directory: {args.output_dir}")

    main(args)
