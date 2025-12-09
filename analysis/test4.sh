#!/bin/bash

# python3 05_evaluate_idk.py --dataname halueval --save_run_name sample_weight_reverse_smooth --dataset all --sft_idk_weight 0.16 --data_eval_method llm --eval_method llm --model_type trained
python3 05_evaluate_idk.py --dataname halueval --save_run_name sample_weight_reverse_smooth --dataset all --sft_idk_weight 0.16 --data_eval_method rouge --data_threshold 0.35 --eval_method llm --model_type trained
# python3 05_evaluate_idk.py --dataname halueval --save_run_name sample_weight_reverse_smooth --dataset all --sft_idk_weight 0.16 --data_eval_method em --eval_method llm --model_type trained