from datasets import load_from_disk
import json

# Convert TriviaQA dataset
print("Converting TriviaQA dataset to JSONL...")
trivia_qa = load_from_disk("./trivia_qa")

for split in trivia_qa.keys():
    print(f"Converting TriviaQA {split} split...")
    with open(f"trivia_qa/trivia_qa_{split}.jsonl", "w", encoding="utf-8") as f:
        for item in trivia_qa[split]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved trivia_qa_{split}.jsonl")

# Convert SQuAD dataset  
print("\nConverting SQuAD dataset to JSONL...")
squad = load_from_disk("./squad")

for split in squad.keys():
    print(f"Converting SQuAD {split} split...")
    with open(f"squad/squad_{split}.jsonl", "w", encoding="utf-8") as f:
        for item in squad[split]:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved squad_{split}.jsonl")

print("\nDone! All datasets have been converted to JSONL format.")