import json
from pathlib import Path

def validate_jsonl(path):
    for i, line in enumerate(open(path)):
        try:
            json.loads(line)
        except Exception as e:
            raise ValueError(f"Invalid JSONL at line {i+1}: {e}")
    print("✓ Data is valid JSONL format.")
