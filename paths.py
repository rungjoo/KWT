"""Shared path conventions so that knowledge estimation, training and evaluation
scripts agree on where every file lives. All paths are resolved relative to the
repository root, so scripts can be launched from any working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
OOD_DIR = DATASET_DIR / "ood"
KNOWLEDGE_DIR = ROOT / "inference"   # knowledge-estimation outputs (base model, multi-sampled)
RESULT_DIR = ROOT / "analysis"       # test-set generations of fine-tuned models
CKPT_DIR = ROOT / "checkpoints"      # fine-tuned model checkpoints

DATANAMES = ["halueval", "medqa", "sciq"]

# ROUGE-L thresholds selected by agreement with human annotation (paper Sec. 4.1)
DEFAULT_ROUGE_THRESHOLD = {"halueval": 0.35, "medqa": 0.6, "sciq": 0.6}

KNOWLEDGE_FILE_PREFIX = "base_model_temp0.7_samples5_fewshot3"
GREEDY_FILE_PREFIX = "base_model_greedy_samples_fewshot3"


def get_model_name(model_path):
    """'meta-llama/Llama-3.2-3B' or '/models/Llama-3.2-3B' -> 'llama-3.2-3b'"""
    return Path(model_path).name.lower()


def split_file(dataname, split):
    """Path to a dataset split ('train' or 'test')."""
    if dataname == "halueval":
        return DATASET_DIR / "halueval" / f"halueval_{split}.jsonl"
    return DATASET_DIR / dataname / f"{split}.jsonl"


def resolve_threshold(dataname, eval_method, threshold=None):
    """Threshold string used in file names: '' for llm/em, e.g. 0.35 for rouge."""
    if eval_method in ("llm", "em"):
        return ""
    if threshold in (None, "", "None"):
        return DEFAULT_ROUGE_THRESHOLD[dataname]
    return float(threshold)


def knowledge_file(dataname, model_name, eval_method, threshold="", split="train", greedy=False):
    """Judged multi-sampled inference file produced by inference/check_answers.py."""
    out_dir = KNOWLEDGE_DIR / dataname / model_name
    if split != "train":
        out_dir = out_dir / split
    prefix = GREEDY_FILE_PREFIX if greedy else KNOWLEDGE_FILE_PREFIX
    return out_dir / f"{prefix}_evaluated_{eval_method}{threshold}.json"


def checkpoint_dir(ckpt_root, dataname, model_name, run_name,
                   eval_method="llm", threshold="", idk_weight=0.16, run_id=0):
    """Directory of a fine-tuned checkpoint."""
    base = Path(ckpt_root) / dataname / model_name / run_name
    if run_name == "sft":
        return base / (f"sft_run{run_id}" if run_id > 0 else "sft")
    if run_name == "seal":
        return base / "seal"
    return base / f"sw_{eval_method}{threshold}_idk{idk_weight}"


def result_stem(run_name, eval_method="llm", threshold="", idk_weight=0.16, run_id=0):
    """File stem of a test-set generation, e.g. 'sample_weight_reverse_smooth_llm_idk0.16'."""
    if run_name == "seal":
        return "seal"
    if run_name == "sft":
        return f"sft_{eval_method}_idk{idk_weight}" + (f"_run{run_id}" if run_id > 0 else "")
    return f"{run_name}_{eval_method}{threshold}_idk{idk_weight}"
