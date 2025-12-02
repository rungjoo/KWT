python3 01_inference_models.py --dataname halueval --base_model ../../model/Qwen2.5-3B --temperature 0.7 --num_samples 5

python3 02_answer_check.py --input_file halueval/qwen2.5-3b/base_model_temp0.7_samples5_fewshot3.jsonl --eval_method em

python3 02_answer_check.py --input_file sciq/qwen2.5-3b/base_model_temp0.7_samples5_fewshot3.jsonl --eval_method rouge --threshold 0.6

python3 02_answer_check.py --input_file halueval/qwen2.5-3b/base_model_temp0.7_samples5_fewshot3.jsonl --eval_method llm