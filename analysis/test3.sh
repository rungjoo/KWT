#!/bin/bash

# python3 01_eval.py --dataname sciq --save_run_name sample_weighted_reverse --sft_idk_weight 0.2 --data_eval_method llm --base_model ../../model/Qwen2.5-3B
# python3 01_eval.py --dataname sciq --save_run_name rtuning_r --sft_idk_weight 1.0 --data_eval_method em --base_model ../../model/Qwen2.5-3B
# python3 01_eval.py --dataname sciq --save_run_name popular --sft_idk_weight 0.0 --data_eval_method llm --base_model ../../model/Qwen2.5-3B
# python3 01_eval2.py --dataname sciq --save_run_name seal --sft_idk_weight 0.0 --data_eval_method llm --base_model ../../model/Qwen2.5-3B

# python3 01_eval.py --dataname sciq --save_run_name sample_weighted_reverse --sft_idk_weight 0.2 --data_eval_method rouge --threshold 0.6 --base_model ../../model/Qwen2.5-3B
# python3 01_eval.py --dataname sciq --save_run_name sample_weighted_reverse --sft_idk_weight 0.2 --data_eval_method em --base_model ../../model/Qwen2.5-3B
# python3 01_eval.py --dataname sciq --save_run_name sample_uniform --sft_idk_weight 1.0 --data_eval_method llm --base_model ../../model/Qwen2.5-3B
# python3 01_eval.py --dataname sciq --save_run_name sample_weighted --sft_idk_weight 1.0 --data_eval_method llm --base_model ../../model/Qwen2.5-3B

python3 02_halueval_answer_check.py --input_file sciq/qwen2.5-3b/sample_weighted_reverse_llm_idk0.2.jsonl --eval_method llm
python3 02_halueval_answer_check.py --input_file sciq/qwen2.5-3b/rtuning_r_em_idk1.0.jsonl --eval_method llm
# python3 02_halueval_answer_check.py --input_file sciq/qwen2.5-3b/popular_llm_idk0.0.jsonl --eval_method llm
python3 02_halueval_answer_check.py --input_file sciq/qwen2.5-3b/seal.jsonl --eval_method llm

python3 02_halueval_answer_check.py --input_file sciq/qwen2.5-3b/sample_weighted_reverse_rouge0.6_idk0.2.jsonl --eval_method llm
python3 02_halueval_answer_check.py --input_file sciq/qwen2.5-3b/sample_weighted_reverse_em_idk0.2.jsonl --eval_method llm
python3 02_halueval_answer_check.py --input_file sciq/qwen2.5-3b/sample_uniform_llm_idk1.0.jsonl --eval_method llm
python3 02_halueval_answer_check.py --input_file sciq/qwen2.5-3b/sample_weighted_llm_idk1.0.jsonl --eval_method llm