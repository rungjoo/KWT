mkdir -p ./halueval/sft_dpo7 && cp -r /mnt/frdata/rungjoo/hall/halu_model/halueval/sft_dpo7/dpo1.0_sft_idk1.0_sft_harder2.0 ./halueval/sft_dpo7/
for task in medqa sciq; do
  mkdir -p ./${task}/sft1 && cp -r /mnt/frdata/rungjoo/hall/halu_model/${task}/sft1/dpo1.0_sft_idk1.0_sft_harder2.0 ./${task}/sft1/
  mkdir -p ./${task}/sft_dpo3 && cp -r /mnt/frdata/rungjoo/hall/halu_model/${task}/sft_dpo3/dpo1.0_sft_idk1.0 ./${task}/sft_dpo3/
  mkdir -p ./${task}/sft_dpo7 && cp -r /mnt/frdata/rungjoo/hall/halu_model/${task}/sft_dpo7/dpo1.0_sft_idk1.0_sft_harder2.0 ./${task}/sft_dpo7/
done
