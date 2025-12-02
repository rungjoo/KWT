# python3 train_sample.py --dataname sciq --epochs 3 --sft_idk_weight 0.2 --save_run_name sample_weighted_reverse --eval_method llm --model_path ../../model/Qwen2.5-3B
# python3 train_sample.py --dataname sciq --epochs 3 --sft_idk_weight 1.0 --save_run_name rtuning_r --eval_method em --model_path ../../model/Qwen2.5-3B
# python3 train_sample.py --dataname sciq --epochs 3 --sft_idk_weight 0.0 --save_run_name popular --eval_method llm --model_path ../../model/Qwen2.5-3B
python3 train_seal.py --dataname sciq --epochs 3 --save_run_name seal --model_path ../../model/Qwen2.5-3B

python3 train_sample.py --dataname sciq --epochs 3 --sft_idk_weight 0.2 --save_run_name sample_weighted_reverse --eval_method rouge --threshold 0.6 --model_path ../../model/Qwen2.5-3B
# python3 train_sample.py --dataname sciq --epochs 3 --sft_idk_weight 0.2 --save_run_name sample_weighted_reverse --eval_method em --model_path ../../model/Qwen2.5-3B
# python3 train_sample.py --dataname sciq --epochs 3 --sft_idk_weight 1.0 --save_run_name sample_uniform --eval_method llm --model_path ../../model/Qwen2.5-3B
# python3 train_sample.py --dataname sciq --epochs 3 --sft_idk_weight 1.0 --save_run_name sample_weighted --eval_method llm --model_path ../../model/Qwen2.5-3B