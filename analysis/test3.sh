#!/bin/bash

python3 05_evaluate_idk.py --dataname halueval --save_run_name sample_weight_reverse_smooth --dataset all --sft_idk_weight 0.16 --data_eval_method llm --eval_method llm --model_type trained
# python3 05_evaluate_idk.py --dataname halueval --save_run_name sample_weight_reverse_smooth --dataset all --sft_idk_weight 0.16 --data_eval_method rouge --data_threshold 0.35 --eval_method llm --model_type trained
# python3 05_evaluate_idk.py --dataname halueval --save_run_name sample_weight_reverse_smooth --dataset all --sft_idk_weight 0.16 --data_eval_method em --eval_method llm --model_type trained

# python3 06_kl_divergence.py --dataname halueval --our_model_name sample_weighted_reverse_llm_idk0.2
# python3 06_kl_divergence.py --dataname medqa --our_model_name sample_weighted_reverse_llm_idk0.2
# python3 06_kl_divergence.py --dataname sciq --our_model_name sample_weighted_reverse_llm_idk0.2

# python3 02_halueval_answer_check.py --input_file sciq/qwen2.5-3b/sample_weighted_reverse_llm_idk0.2.jsonl --eval_method llm
# python3 02_halueval_answer_check.py --input_file sciq/qwen2.5-3b/rtuning_r_em_idk1.0.jsonl --eval_method llm
# python3 02_halueval_answer_check.py --input_file sciq/qwen2.5-3b/popular_llm_idk0.0.jsonl --eval_method llm
# python3 02_halueval_answer_check.py --input_file sciq/qwen2.5-3b/seal.jsonl --eval_method llm

# python3 02_halueval_answer_check.py --input_file sciq/qwen2.5-3b/sample_weighted_reverse_rouge0.6_idk0.2.jsonl --eval_method llm
# python3 02_halueval_answer_check.py --input_file sciq/qwen2.5-3b/sample_weighted_reverse_em_idk0.2.jsonl --eval_method llm
# python3 02_halueval_answer_check.py --input_file sciq/qwen2.5-3b/sample_uniform_llm_idk1.0.jsonl --eval_method llm
# python3 02_halueval_answer_check.py --input_file sciq/qwen2.5-3b/sample_weighted_llm_idk1.0.jsonl --eval_method llm