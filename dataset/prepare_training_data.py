import json
import random
from typing import Dict, List

def load_squad_data(filepath: str) -> List[Dict]:
    """Load and format SQuAD data"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            # Format for QA task with question/answer structure
            formatted_item = {
                "question": f"Context: {item['context']}\n\nQuestion: {item['question']}",
                "answer": item['answers']['text'][0] if item['answers']['text'] else ""
            }
            data.append(formatted_item)
    return data

def load_trivia_qa_data(filepath: str) -> List[Dict]:
    """Load and format TriviaQA data"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            # Use the first alias as the answer
            answer = ""
            if item.get('answer') and item['answer'].get('aliases'):
                answer = item['answer']['aliases'][0]
            
            formatted_item = {
                "question": item['question'],
                "answer": answer
            }
            data.append(formatted_item)
    return data

def merge_and_save_data():
    """Merge SQuAD and TriviaQA data and save as training and validation data"""
    # Load training data
    print("Loading SQuAD training data...")
    squad_train = load_squad_data("squad/squad_train.jsonl")
    print(f"Loaded {len(squad_train)} SQuAD training examples")
    
    print("Loading TriviaQA training data...")
    trivia_train = load_trivia_qa_data("trivia_qa/trivia_qa_train.jsonl")
    print(f"Loaded {len(trivia_train)} TriviaQA training examples")
    
    # Load validation data
    print("\nLoading SQuAD validation data...")
    squad_val = load_squad_data("squad/squad_validation.jsonl")
    print(f"Loaded {len(squad_val)} SQuAD validation examples")
    
    print("Loading TriviaQA validation data...")
    trivia_val = load_trivia_qa_data("trivia_qa/trivia_qa_validation.jsonl")
    print(f"Loaded {len(trivia_val)} TriviaQA validation examples")
    
    # Combine datasets
    train_data = squad_train + trivia_train
    val_data = squad_val + trivia_val
    
    # Shuffle the data
    random.seed(42)
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    print(f"\nTotal training examples: {len(train_data)}")
    print(f"Total validation examples: {len(val_data)}")
    
    # Save merged training data
    with open("merged_train_data.jsonl", 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print("Saved merged training data to merged_train_data.jsonl")
    
    # Save merged validation data
    with open("merged_val_data.jsonl", 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print("Saved merged validation data to merged_val_data.jsonl")

if __name__ == "__main__":
    merge_and_save_data()