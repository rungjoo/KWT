#!/bin/bash

python3 01_eval.py --dataname halueval --save_run_name sample_weight_reverse_smooth --sft_idk_weight 0.16 --data_eval_method rouge --threshold 0.35 --base_model ../../model/Llama-3.2-3B
python3 01_eval.py --dataname medqa --save_run_name sample_weight_reverse_smooth --sft_idk_weight 0.16 --data_eval_method rouge --threshold 0.6 --base_model ../../model/Llama-3.2-3B
python3 01_eval.py --dataname sciq --save_run_name sample_weight_reverse_smooth --sft_idk_weight 0.16 --data_eval_method rouge --threshold 0.6 --base_model ../../model/Llama-3.2-3B

python3 02_halueval_answer_check.py --input_file halueval/llama-3.2-3b/sample_weight_reverse_smooth_rouge0.35_idk0.16.jsonl --eval_method llm
python3 02_halueval_answer_check.py --input_file medqa/llama-3.2-3b/sample_weight_reverse_smooth_rouge0.6_idk0.16.jsonl --eval_method llm
python3 02_halueval_answer_check.py --input_file sciq/llama-3.2-3b/sample_weight_reverse_smooth_rouge0.6_idk0.16.jsonl --eval_method llm
