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
# 1) 데이터 로딩 (SFT/DPO 구분)
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

        if base_match and self_sft_match:
            item['response'] = right_answer
            item['loss_type'] = 'sft'
        elif not base_match and not self_sft_match:
            item['response'] = "I don't know"
            item['loss_type'] = 'sft'
        elif base_match and not self_sft_match:
            item['chosen'] = base_answer
            item['rejected'] = self_sft_answer
            item['loss_type'] = 'dpo'
        elif not base_match and self_sft_match:
            item['chosen'] = self_sft_answer
            item['rejected'] = base_answer
            item['loss_type'] = 'dpo'
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
    prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length, return_tensors=None)["input_ids"]
    prompt_len = len(prompt_ids)

    ex = {
        "loss_type": loss_type,
        "input_ids": None,
        "labels": None,
        "chosen_input_ids": None,
        "rejected_input_ids": None,
        "prompt_len": None,
    }

    if loss_type == "sft":
        full_text = prompt + " " + item['response']
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
    
    else: # loss_type == "dpo"
        # Tokenize chosen response
        chosen_text = prompt + " " + item['chosen']
        chosen_tokenized = tokenizer(chosen_text, truncation=True, max_length=max_length, return_tensors=None)
        chosen_input_ids = chosen_tokenized["input_ids"]
        
        # Tokenize rejected response
        rejected_text = prompt + " " + item['rejected']
        rejected_tokenized = tokenizer(rejected_text, truncation=True, max_length=max_length, return_tensors=None)
        rejected_input_ids = rejected_tokenized["input_ids"]

        ex["chosen_input_ids"]   = chosen_input_ids
        ex["rejected_input_ids"] = rejected_input_ids
        ex["prompt_len"]         = prompt_len
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
# 3) Collator: 배치에서 SFT/DPO를 분리 패딩 (배치처리하면서 패딩하는 코드라 보면 되는 듯)
######################################
@dataclass
class MixedSftDpoCollator:
    tokenizer: AutoTokenizer

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        sft_items = [b for b in batch if b["loss_type"] == "sft"]
        dpo_items = [b for b in batch if b["loss_type"] == "dpo"]

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
        
        if len(dpo_items) > 0:
            chosen_input_ids = [x['chosen_input_ids'] for x in dpo_items]
            rejected_input_ids = [x['rejected_input_ids'] for x in dpo_items]

            chosen_input, chosen_attn = self.padding_input(chosen_input_ids)
            rejected_input, rejected_attn = self.padding_input(rejected_input_ids)

            prompt_lens = torch.tensor([x["prompt_len"] for x in dpo_items], dtype=torch.long)

            collated["dpo"] = {
                "chosen_input_ids": chosen_input,
                "chosen_attention_mask": chosen_attn,
                "rejected_input_ids": rejected_input,
                "rejected_attention_mask": rejected_attn,
                "prompt_len": prompt_lens,
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
    def __init__(self, *args, ref_model=None, dpo_beta=0.1, sft_weight=1.0, dpo_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        assert ref_model is not None, "ref_model must be provided for DPO."
        self.ref_model = ref_model
        self.dpo_beta = dpo_beta
        self.sft_weight = sft_weight
        self.dpo_weight = dpo_weight
        # ref_model은 고정
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()
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
            total_loss = total_loss + self.sft_weight * sft_loss
            logs["sft/loss"] = sft_loss.detach().item()

        # DPO loss
        if "dpo" in inputs:
            d = self._prepare_inputs(inputs["dpo"])
            d_loss, d_logs = dpo_loss(
                policy_model=model,
                ref_model=self.ref_model,
                chosen_input_ids=d["chosen_input_ids"],
                chosen_attention_mask=d["chosen_attention_mask"],
                rejected_input_ids=d["rejected_input_ids"],
                rejected_attention_mask=d["rejected_attention_mask"],
                prompt_len=d["prompt_len"],
                beta=self.dpo_beta,
            )    
            total_loss = total_loss + self.dpo_weight * d_loss
            logs["dpo/loss"] = d_loss.detach().item()
            logs.update(d_logs)

        self.log(logs)
        return (total_loss, None) if return_outputs else total_loss

######################################
# 4-1) DPO loss 계산 유틸
######################################        
def dpo_loss(
    policy_model,
    ref_model,
    chosen_input_ids, chosen_attention_mask,
    rejected_input_ids, rejected_attention_mask,
    prompt_len,
    beta=0.1,
):
    """
    L = - log σ( β * [ (logπθ(y+)-logπθ(y-)) - (logπref(y+)-logπref(y-)) ] )
    """
    # 정책모델 로그확률
    policy_chosen = sequence_logprobs(policy_model, chosen_input_ids, chosen_attention_mask, prompt_len)
    policy_rejected = sequence_logprobs(policy_model, rejected_input_ids, rejected_attention_mask, prompt_len)
    # 참고모델 로그확률(gradient X)
    with torch.no_grad():
        ref_chosen = sequence_logprobs(ref_model, chosen_input_ids, chosen_attention_mask, prompt_len)
        ref_rejected = sequence_logprobs(ref_model, rejected_input_ids, rejected_attention_mask, prompt_len)

    policy_diff = policy_chosen - policy_rejected
    ref_diff = ref_chosen - ref_rejected
    advantage = policy_diff - ref_diff
    loss = -torch.nn.functional.logsigmoid(beta * advantage).mean()
    # 모니터링용 metric도 함께 반환

    # chosen 샘플을 더 큰 reward로 판단했다는 의미로 accuracy
    with torch.no_grad():
        accuracy = (policy_diff > 0).float().mean()
    return loss, {
        "dpo/policy_chosen": policy_chosen.mean().item(),
        "dpo/policy_rejected": policy_rejected.mean().item(),
        "dpo/policy_diff": policy_diff.mean().item(),
        "dpo/ref_diff": ref_diff.mean().item(),
        "dpo/accuracy": accuracy.item(),
        "dpo/advantage": advantage.mean().item(),
    }

def sequence_logprobs(model, input_ids, attention_mask, prompt_len: torch.Tensor):
    """
    모델의 토큰별 로짓 -> 로그확률 -> '답변 구간'(prompt_len 이후)만 합산한 수열 로그확률.
    input_ids: (B, T)
    prompt_len: (B,)  각 샘플의 프롬프트 길이
    반환: (B,)
    """
    with torch.no_grad() if not model.training else torch.enable_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, T, V)
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    # 답변 구간 마스크 만들기 (프롬프트 다음 토큰부터)
    B, Tm1 = shift_labels.size()
    device = input_ids.device
    token_index = torch.arange(Tm1, device=device).unsqueeze(0).expand(B, -1)  # (B, T-1)
    # 각 배치별 시작 인덱스
    # prompt_len 은 원문 시퀀스 기준, shift 후에도 동일하게 적용(답변 시작 토큰이 shift_labels의 인덱스 prompt_len-1 이 됨)
    # 안전하게: 답변 마스크: token_index >= (prompt_len - 1)
    start_idx = (prompt_len - 1).clamp(min=0).unsqueeze(1)
    answer_mask = (token_index >= start_idx).long()

    # attention mask도 shift
    shift_attn = attention_mask[:, 1:]
    valid_mask = shift_attn * answer_mask  # (B, T-1)

    # cross-entropy: -log p(y_t | x, y_<t)
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_logp = torch.gather(log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

    token_logp = token_logp * valid_mask  # prompt/패딩 제외
    seq_logp = token_logp.sum(dim=-1)     # (B,)
    return seq_logp


def main(args):
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "model_path": args.model_path,
            "train_data_path": args.train_data_path,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "grad_accum": args.grad_accum,
            "dpo_beta": args.dpo_beta,
            "sft_weight": args.sft_weight,
            "dpo_weight": args.dpo_weight,
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

    # reference model: 초기 모델 스냅샷
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
    )
    ref_model.eval()    

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
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        dpo_beta=args.dpo_beta,
        sft_weight=args.sft_weight,
        dpo_weight=args.dpo_weight,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training finished.")    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model answers using LLM')
    parser.add_argument('--train_data_path', type=str, default="../dataset_llama_3.2-3b/merged/merged_evaluated.json")
    parser.add_argument('--model_path', type=str, default="../../model/Llama-3.2-3B")
    parser.add_argument('--output_dir', type=str, default="./checkpoints")
    parser.add_argument('--max_length', type=int, default=1024)

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)

    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--save_steps', type=int, default=200)

    # DPO 하이퍼파라미터 및 loss 가중치
    parser.add_argument('--dpo_beta', type=float, default=0.1)
    parser.add_argument('--sft_weight', type=float, default=1.0)
    parser.add_argument('--dpo_weight', type=float, default=1.0)

    # Wandb configuration
    parser.add_argument('--wandb_project', type=str, default="hall-sft-dpo", help='WandB project name')
    parser.add_argument('--wandb_run_name', type=str, default="test", help='WandB run name (optional)')

    args = parser.parse_args()    
    main(args)