for dir in halueval medqa sciq; do
  cd /mnt/fr20tb/rungjoo/hall/analysis/$dir/llama_3.2-3b

  # 1️⃣ 임시 이름으로 바꾸기 (충돌 방지)
  for f in *instruct_idk*; do
    mv "$f" "${f/instruct_idk/instruct_tmp}"
  done

  # 2️⃣ no_idk → idk
  for f in *instruct_no_idk*; do
    mv "$f" "${f/instruct_no_idk/instruct_idk}"
  done

  # 3️⃣ tmp → no_idk
  for f in *instruct_tmp*; do
    mv "$f" "${f/instruct_tmp/instruct_no_idk}"
  done
done
