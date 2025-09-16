import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

class MixedLossTrainer(Trainer):
    """Custom trainer that applies SFT or DPO loss based on data type"""
    
    def __init__(self, *args, beta: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta  # DPO beta parameter
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss based on data type:
        - SFT loss for both_correct and both_wrong
        - DPO loss for only_base_correct and only_sft_correct
        """
        loss_type = inputs.pop("loss_type", None)
        
        if loss_type in ["both_correct", "both_wrong"]:
            # SFT Loss
            outputs = model(**inputs)
            loss = outputs.loss
        else:
            # DPO Loss
            chosen_input_ids = inputs.pop("chosen_input_ids")
            rejected_input_ids = inputs.pop("rejected_input_ids")
            chosen_attention_mask = inputs.pop("chosen_attention_mask")
            rejected_attention_mask = inputs.pop("rejected_attention_mask")
            chosen_labels = inputs.pop("chosen_labels")
            rejected_labels = inputs.pop("rejected_labels")
            
            # Get logits for chosen and rejected
            with torch.no_grad():
                chosen_outputs = model(
                    input_ids=chosen_input_ids,
                    attention_mask=chosen_attention_mask,
                    labels=chosen_labels
                )
                rejected_outputs = model(
                    input_ids=rejected_input_ids,
                    attention_mask=rejected_attention_mask,
                    labels=rejected_labels
                )
            
            # Calculate DPO loss
            chosen_logps = -chosen_outputs.loss
            rejected_logps = -rejected_outputs.loss
            
            # DPO loss: -log(sigmoid(beta * (log p(chosen) - log p(rejected))))
            loss = -F.logsigmoid(self.beta * (chosen_logps - rejected_logps)).mean()
            
            outputs = type('obj', (object,), {'loss': loss})()
        
        return (loss, outputs) if return_outputs else loss

@dataclass
class DataCollatorForMixedTraining:
    """Data collator for mixed SFT and DPO training"""
    tokenizer: AutoTokenizer
    max_length: int = 512
    
    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = {}
        
        # Separate by loss type
        sft_instances = [x for x in instances if x['loss_type'] in ['both_correct', 'both_wrong']]
        dpo_instances = [x for x in instances if x['loss_type'] in ['only_base_correct', 'only_sft_correct']]
        
        if sft_instances:
            # Process SFT instances
            input_ids = []
            labels = []
            attention_masks = []
            
            for instance in sft_instances:
                input_ids.append(instance["input_ids"])
                labels.append(instance["labels"])
                attention_masks.append([1] * len(instance["input_ids"]))
            
            # Pad sequences
            max_len = max(len(ids) for ids in input_ids)
            
            for i in range(len(input_ids)):
                padding_length = max_len - len(input_ids[i])
                input_ids[i] = input_ids[i] + [self.tokenizer.pad_token_id] * padding_length
                labels[i] = labels[i] + [-100] * padding_length
                attention_masks[i] = attention_masks[i] + [0] * padding_length
            
            batch["input_ids"] = torch.tensor(input_ids)
            batch["labels"] = torch.tensor(labels)
            batch["attention_mask"] = torch.tensor(attention_masks)
            batch["loss_type"] = torch.tensor([0] * len(sft_instances))  # 0 for SFT
            
        if dpo_instances:
            # Process DPO instances
            chosen_input_ids = []
            rejected_input_ids = []
            chosen_labels = []
            rejected_labels = []
            chosen_attention_masks = []
            rejected_attention_masks = []
            
            for instance in dpo_instances:
                chosen_input_ids.append(instance["chosen_input_ids"])
                rejected_input_ids.append(instance["rejected_input_ids"])
                chosen_labels.append(instance["chosen_labels"])
                rejected_labels.append(instance["rejected_labels"])
                chosen_attention_masks.append([1] * len(instance["chosen_input_ids"]))
                rejected_attention_masks.append([1] * len(instance["rejected_input_ids"]))
            
            # Pad sequences
            max_chosen_len = max(len(ids) for ids in chosen_input_ids)
            max_rejected_len = max(len(ids) for ids in rejected_input_ids)
            
            for i in range(len(chosen_input_ids)):
                # Pad chosen
                padding_length = max_chosen_len - len(chosen_input_ids[i])
                chosen_input_ids[i] = chosen_input_ids[i] + [self.tokenizer.pad_token_id] * padding_length
                chosen_labels[i] = chosen_labels[i] + [-100] * padding_length
                chosen_attention_masks[i] = chosen_attention_masks[i] + [0] * padding_length
                
                # Pad rejected
                padding_length = max_rejected_len - len(rejected_input_ids[i])
                rejected_input_ids[i] = rejected_input_ids[i] + [self.tokenizer.pad_token_id] * padding_length
                rejected_labels[i] = rejected_labels[i] + [-100] * padding_length
                rejected_attention_masks[i] = rejected_attention_masks[i] + [0] * padding_length
            
            batch["chosen_input_ids"] = torch.tensor(chosen_input_ids)
            batch["rejected_input_ids"] = torch.tensor(rejected_input_ids)
            batch["chosen_labels"] = torch.tensor(chosen_labels)
            batch["rejected_labels"] = torch.tensor(rejected_labels)
            batch["chosen_attention_mask"] = torch.tensor(chosen_attention_masks)
            batch["rejected_attention_mask"] = torch.tensor(rejected_attention_masks)
            batch["loss_type"] = torch.tensor([1] * len(dpo_instances))  # 1 for DPO
        
        return batch

def load_merged_data(file_path: str) -> Dict:
    """Load the merged evaluated JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_prompt(question: str, knowledge: str) -> str:
    """Create prompt from question and knowledge"""
    if knowledge:
        return f"Knowledge: {knowledge}\n\nQuestion: {question}\n\nAnswer:"
    return f"Question: {question}\n\nAnswer:"

def tokenize_for_sft(item: Dict, tokenizer: AutoTokenizer, max_length: int = 512) -> Dict:
    """Tokenize data for SFT training"""
    prompt = create_prompt(item['question'], item['knowledge'])
    full_text = prompt + " " + item['answer']
    
    # Tokenize full text
    tokenized = tokenizer(full_text, truncation=True, max_length=max_length, return_tensors=None)
    input_ids = tokenized["input_ids"]
    
    # Find where answer starts
    prompt_tokens = tokenizer(prompt, truncation=True, max_length=max_length, return_tensors=None)["input_ids"]
    answer_start_pos = len(prompt_tokens)
    
    # Create labels: -100 for prompt, actual tokens for answer
    labels = [-100] * answer_start_pos + input_ids[answer_start_pos:]
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_type": item['type']
    }

def tokenize_for_dpo(item: Dict, tokenizer: AutoTokenizer, max_length: int = 512) -> Dict:
    """Tokenize data for DPO training"""
    prompt = create_prompt(item['question'], item['knowledge'])
    
    # Tokenize chosen response
    chosen_text = prompt + " " + item['chosen']
    chosen_tokenized = tokenizer(chosen_text, truncation=True, max_length=max_length, return_tensors=None)
    chosen_input_ids = chosen_tokenized["input_ids"]
    
    # Tokenize rejected response
    rejected_text = prompt + " " + item['rejected']
    rejected_tokenized = tokenizer(rejected_text, truncation=True, max_length=max_length, return_tensors=None)
    rejected_input_ids = rejected_tokenized["input_ids"]
    
    # Find where answer starts
    prompt_tokens = tokenizer(prompt, truncation=True, max_length=max_length, return_tensors=None)["input_ids"]
    answer_start_pos = len(prompt_tokens)
    
    # Create labels
    chosen_labels = [-100] * answer_start_pos + chosen_input_ids[answer_start_pos:]
    rejected_labels = [-100] * answer_start_pos + rejected_input_ids[answer_start_pos:]
    
    return {
        "chosen_input_ids": chosen_input_ids,
        "rejected_input_ids": rejected_input_ids,
        "chosen_labels": chosen_labels,
        "rejected_labels": rejected_labels,
        "loss_type": item['type']
    }

def prepare_dataset(data: Dict, tokenizer: AutoTokenizer) -> List[Dict]:
    """Prepare dataset with all 4 types of training data"""
    processed_data = []
    
    for item in data['results']:
        base_match = item['base_model']['match']
        sft_match = item['self_sft_model']['match']
        
        if base_match and sft_match:
            # Both correct - SFT with right answer
            sft_item = {
                'question': item['question'],
                'knowledge': item['knowledge'],
                'answer': item['right_answer'],
                'type': 'both_correct'
            }
            processed_data.append(tokenize_for_sft(sft_item, tokenizer))
            
        elif not base_match and not sft_match:
            # Both wrong - SFT with "I don't know"
            sft_item = {
                'question': item['question'],
                'knowledge': item['knowledge'],
                'answer': "I don't know",
                'type': 'both_wrong'
            }
            processed_data.append(tokenize_for_sft(sft_item, tokenizer))
            
        elif base_match and not sft_match:
            # Only base correct - DPO
            dpo_item = {
                'question': item['question'],
                'knowledge': item['knowledge'],
                'chosen': item['right_answer'],
                'rejected': item['self_sft_model']['filtered_model_answer'],
                'type': 'only_base_correct'
            }
            processed_data.append(tokenize_for_dpo(dpo_item, tokenizer))
            
        elif not base_match and sft_match:
            # Only SFT correct - DPO
            dpo_item = {
                'question': item['question'],
                'knowledge': item['knowledge'],
                'chosen': item['right_answer'],
                'rejected': item['base_model']['filtered_model_answer'],
                'type': 'only_sft_correct'
            }
            processed_data.append(tokenize_for_dpo(dpo_item, tokenizer))
    
    return processed_data

def main():
    # Configuration
    model_name = "meta-llama/Llama-3.2-3B"
    output_dir = "./Llama-3.2-3B-Hallucination-Mixed"
    data_path = "../dataset_llama_3.2-3b/merged/merged_evaluated.json"
    
    # Load tokenizer and model
    print(f"Loading model and tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Load and prepare data
    print(f"Loading data from {data_path}...")
    data = load_merged_data(data_path)
    
    print("Preparing dataset...")
    processed_data = prepare_dataset(data, tokenizer)
    
    # Count data types
    type_counts = {}
    for item in processed_data:
        loss_type = item.get('loss_type', item.get('type'))
        type_counts[loss_type] = type_counts.get(loss_type, 0) + 1
    
    print("\n=== Data Statistics ===")
    print(f"Both correct (SFT): {type_counts.get('both_correct', 0)}")
    print(f"Both wrong (SFT): {type_counts.get('both_wrong', 0)}")
    print(f"Only base correct (DPO): {type_counts.get('only_base_correct', 0)}")
    print(f"Only SFT correct (DPO): {type_counts.get('only_sft_correct', 0)}")
    print(f"Total samples: {len(processed_data)}")
    
    # Create dataset
    train_dataset = Dataset.from_list(processed_data)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        optim="adamw_torch",
        learning_rate=2e-5,
        warmup_steps=100,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="no",
        fp16=False,
        bf16=True,
        push_to_hub=False,
        report_to=["tensorboard"],
        logging_dir=f"{output_dir}/logs",
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False,
        group_by_length=False,  # Don't group by length for mixed training
        remove_unused_columns=False,
    )
    
    # Create custom trainer
    trainer = MixedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMixedTraining(tokenizer=tokenizer),
        beta=0.1,  # DPO beta parameter
    )
    
    # Print training info
    print("\n=== Training Configuration ===")
    print(f"Model: {model_name}")
    print(f"Output directory: {output_dir}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of epochs: {training_args.num_train_epochs}")
    print(f"Batch size: {training_args.per_device_train_batch_size}")
    print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"DPO Beta: 0.1")
    print("==============================\n")
    
    # Start training
    print("Starting mixed training...")
    trainer.train()
    
    # Save final model
    print(f"Saving final model to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("Training complete!")

if __name__ == "__main__":
    main()