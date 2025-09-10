# SFT 모델
squad, trivia_qa 데이터를 merged_train, merged_val 형태로 데이터 저장함
해당 데이터로 base을 SFT 학습시킴

# target SFT 모델
우리가 원래 base 모델에 학습시키려고 하는데 target SFT 데이터가 있다고 했을때, 이 데이터를 분류작업을 한다.
해당 target 데이터는 new_sft 데이터라고 보면됨
위에서 학습시킨 SFT 모델과 BASE 모델을 이용해서 데이터 분류함