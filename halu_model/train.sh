# python3 train_sft_dpo.py --dataname halueval --epochs 3 --save_steps 250 --dpo_weight 1.0 --sft_idk_weight 1.0 --sft_harder_weight 1.5 --save_run_name sft1
# python3 train_sft_dpo.py --dataname medqa --epochs 3 --save_steps 250 --dpo_weight 1.0 --sft_idk_weight 1.0 --sft_harder_weight 1.5 --save_run_name sft1
# python3 train_sft_dpo.py --dataname sciq --epochs 3 --save_steps 250 --dpo_weight 1.0 --sft_idk_weight 1.0 --sft_harder_weight 1.5 --save_run_name sft1

python3 train_sft_dpo.py --dataname halueval --epochs 3 --save_steps 250 --dpo_weight 1.0 --sft_idk_weight 1.0 --sft_harder_weight 2.0 --save_run_name sft1_s --model_path ../ref_model/Llama-3.2-3B-SFT
python3 train_sft_dpo.py --dataname medqa --epochs 3 --save_steps 250 --dpo_weight 1.0 --sft_idk_weight 1.0 --sft_harder_weight 2.0 --save_run_name sft1_s --model_path ../ref_model/Llama-3.2-3B-SFT
python3 train_sft_dpo.py --dataname sciq --epochs 3 --save_steps 250 --dpo_weight 1.0 --sft_idk_weight 1.0 --sft_harder_weight 2.0 --save_run_name sft1_s --model_path ../ref_model/Llama-3.2-3B-SFT

# python3 train_sft_dpo.py --dataname halueval --epochs 3 --save_steps 250 --dpo_weight 1.0 --sft_idk_weight 1.0 --save_run_name sft_dpo4
# python3 train_sft_dpo.py --dataname medqa --epochs 3 --save_steps 250 --dpo_weight 1.0 --sft_idk_weight 1.0 --save_run_name sft_dpo4
# python3 train_sft_dpo.py --dataname sciq --epochs 3 --save_steps 250 --dpo_weight 1.0 --sft_idk_weight 1.0 --save_run_name sft_dpo4

# python3 train_sft_dpo.py --dataname halueval --epochs 3 --save_steps 250 --dpo_weight 0.5 --sft_idk_weight 1.0 --save_run_name sft_dpo3
# python3 train_sft_dpo.py --dataname medqa --epochs 3 --save_steps 250 --dpo_weight 0.5 --sft_idk_weight 1.0 --save_run_name sft_dpo3
# python3 train_sft_dpo.py --dataname sciq --epochs 3 --save_steps 250 --dpo_weight 0.5 --sft_idk_weight 1.0 --save_run_name sft_dpo3

# python3 train_sft_dpo.py --dataname halueval --epochs 3 --save_steps 250 --dpo_weight 0.1 --sft_idk_weight 1.0 --save_run_name sft_dpo3
# python3 train_sft_dpo.py --dataname medqa --epochs 3 --save_steps 250 --dpo_weight 0.1 --sft_idk_weight 1.0 --save_run_name sft_dpo3
# python3 train_sft_dpo.py --dataname sciq --epochs 3 --save_steps 250 --dpo_weight 0.1 --sft_idk_weight 1.0 --save_run_name sft_dpo3