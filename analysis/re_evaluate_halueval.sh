python3 01_eval.py --dataname halueval --save_run_name sft3
python3 01_eval.py --dataname medqa --save_run_name sft3

python3 02_halueval_answer_check.py --input_file halueval/llama_3.2-3b/sft3.jsonl
python3 02_halueval_answer_check.py --input_file medqa/llama_3.2-3b/sft3.jsonl