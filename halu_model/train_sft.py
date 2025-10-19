import os
import torch
import json
import math
import argparse
import wandb
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

######################################
# 1) 데이터 로딩 (SFT로만 사용)
######################################
def load_data(filepath):    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data_type = []
    for result in data['results']:
        question = result['question']
        right_answer = result['right_answer']
        base_answer = result['base_model']['filtered_model_answer']
        self_sft_answer = result['self_sft_model']['filtered_model_answer']

        item = {}
        item['question'] = question

        base_match = result['base_model']['match']
        self_sft_match = result['self_sft_model']['match']

        item['response'] = right_answer
        data_type.append(item)
    return data_type

######################################
# 2) 프롬프트/토크나이즈
######################################
def create_prompt(question):
    return f"Question: {question}\n\nAnswer:"

def preprocess_tokenize(item, tokenizer, max_length=1024):
    question = item['question']
    prompt = create_prompt(question)

    ex = {
        "input_ids": None,
        "labels": None
    }

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

    ex["input_ids"] = input_ids
    ex["labels"]    = labels
    return ex
    
# {'question': 'Hans Kammler was the last in Germany to be appointed to the paramilitary rank first created in what year?',
#  'knowledge': ' As an SS officer, he was the last person in Nazi Germany to be appointed to the rank of "SS-Obergruppenführer".Obergruppenführer (] , "senior group leader") was a Nazi Party paramilitary rank that was first created in 1932 as a rank of the "Sturmabteilung" (SA), and adopted by the "Schutzstaffel" (SS) one year later.',
#  'right_answer': '1932',
#  'hallucinated_answer': 'Hans Kammler was the last person appointed to the Nazi Party paramilitary rank in 1933.',
#  'base_model': {'model_answer': '1932\n\nQuestion: What sports complex was the 2014 MLS Cup match in California?\nAnswer: StubHub Center\n\nQuestion: For which number novel did the author of "The Secret History" win the Pulitzer Price for fiction?\nAnswer:',
#   'filtered_model_answer': '1932',
#   'evaluation': 'match',
#   'match': True},
#  'instruct_model': {'model_answer': '1933.\n\nExplanation: Hans Kammler was a German SS officer who was appointed to the rank of SS-Sturmbannführer in 1933, which was the highest rank in the SS at that time. The SS was a',
#   'filtered_model_answer': '1933.',
#   'evaluation': 'mismatch',
#   'match': False},
#  'self_sft_model': {'model_answer': '1933',
#   'filtered_model_answer': '1933',
#   'evaluation': 'mismatch',
#   'match': False},
#  'agreement': 'only_base_correct'}   
 
######################################
# 3) Collator: 배치에서 SFT 데이터 패딩 (배치처리하면서 패딩하는 코드라 보면 되는 듯)
######################################
@dataclass
class MixedSftDpoCollator:
    tokenizer: AutoTokenizer

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        sft_items = batch

        collated: Dict[str, Any] = {}

        if len(sft_items) > 0:
            sft_input_ids = [x['input_ids'] for x in sft_items]
            sft_labels = [x['labels'] for x in sft_items]
            sft_input, sft_attn = self.padding_input(sft_input_ids)
            sft_labels = self.padding_label(sft_labels, max_len=sft_input.size(1))
            collated["sft"] = {
                "input_ids": sft_input,
                "attention_mask": sft_attn,
                "labels": sft_labels,
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
class MixedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    # 여기서 return_outputs=True은 평가중일떄 예측일 필요한 경우를 대비해서임
    # 사실 우리는 return_outputs=False인 경우만 고려하면 되긴함
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        total_loss = 0.0
        logs = {}
        device = next(model.parameters()).device  # 모델이 올라간 디바이스 확인

        # SFT loss (표준 LM loss)
        if "sft" in inputs:
            sft_batch = self._prepare_inputs(inputs["sft"])
            outputs = model(**sft_batch)
            sft_loss = outputs.loss
            total_loss = total_loss + sft_loss
            logs["sft/loss"] = sft_loss.detach().item()

        self.log(logs)
        return (total_loss, None) if return_outputs else total_loss

def main(args):
    # Initialize wandb
    wandb_run_name = f"{args.dataname}_{args.save_run_name}-{args.epochs}"
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
        tokenizer.pad_token = tokenizer.eos_token  # 안전하게 eos를 pad로 사용
        print("Set: tokenizer padding")

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
    train_data = load_data(train_data_path)
    print(f"Loaded {len(train_data)} training examples")

    train_tokenized = []
    for ex in train_data:
        train_tokenized.append(preprocess_tokenize(ex, tokenizer, max_length=args.max_length))
    train_ds = Dataset.from_list(train_tokenized)

    collator = MixedSftDpoCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps= args.logging_steps,
        save_steps= args.save_steps,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),  # A100/H100면 bf16, 아니면 자동으로 fp32 사용
        remove_unused_columns=False,     # collator가 dict 구조를 유지하도록
        report_to="wandb",
    )    

    trainer = MixedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training finished.")    

if __name__ == "__main__":
    # python3 train_sft.py --dataname halueval --epochs 3 --save_steps 250 --save_run_name sft
    # python3 train_sft.py --dataname medqa --epochs 3 --save_steps 250 --save_run_name sft
    # python3 train_sft.py --dataname sciq --epochs 3 --save_steps 250 --save_run_name sft
    parser = argparse.ArgumentParser(description='Evaluate model answers using LLM')
    parser.add_argument('--dataname', type=str, required=True, help='Dataset name (e.g., halueval, medqa)')
    parser.add_argument('--model_path', type=str, default="../../model/Llama-3.2-3B")
    parser.add_argument('--max_length', type=int, default=1024)

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)

    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--save_steps', type=int, default=200)

    # Wandb configuration
    parser.add_argument('--wandb_project', type=str, default="hall-sft-dpo", help='WandB project name')
    parser.add_argument('--save_run_name', type=str, default="sft", help='WandB run name (optional)')

    args = parser.parse_args()

    # Set train_data_path and output_dir based on dataname
    args.train_data_path = f"../dataset_type/{args.dataname}/merged/merged_evaluated.json"
    args.output_dir = f"./{args.dataname}/{args.save_run_name}"

    print(f"Dataset: {args.dataname}")
    print(f"Train data path: {args.train_data_path}")
    print(f"Output directory: {args.output_dir}")

    main(args)