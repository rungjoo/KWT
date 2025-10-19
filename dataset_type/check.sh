python3 01_inference_models.py --dataname sciq --model base
python3 02_answer_check.py --input_file sciq/base_model_results.jsonl
python3 03_merge_evaluated_results.py --dataname sciq