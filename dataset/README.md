# Dataset

Datasets for training and evaluation.

## Folder Structure

```
dataset/
├── halueval/          # HaluEval dataset
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── medqa/             # MedQA dataset
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
└── sciq/              # SciQ dataset
    ├── train.jsonl
    ├── val.jsonl
    └── test.jsonl
```

## Dataset Description

| Dataset | Domain | Description |
|---------|--------|-------------|
| HaluEval | General | QA dataset for hallucination evaluation |
| MedQA | Medical | Medical knowledge QA dataset |
| SciQ | Science | Science knowledge QA dataset |

## Data Format

Each JSONL file record format:

```json
{
  "question": "Question content",
  "answer": "Correct answer",
  "knowledge": "Related knowledge (optional)",
  "support": "Supporting information (optional)"
}
```

## Field Description

| Field | Description | Required |
|-------|-------------|----------|
| `question` | Question text | Yes |
| `answer` / `right_answer` / `correct_answer` | Correct answer | Yes |
| `knowledge` / `support` | Background knowledge | No |

## Usage Example

```python
import json

def load_dataset(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# Load data
train_data = load_dataset('dataset/halueval/train.jsonl')
test_data = load_dataset('dataset/halueval/test.jsonl')
```

## Data Splits

| Split | Purpose |
|-------|---------|
| `train.jsonl` | Model training |
| `val.jsonl` | Validation and hyperparameter tuning |
| `test.jsonl` | Final evaluation |
