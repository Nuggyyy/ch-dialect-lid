#!/usr/bin/env python3
"""
Extract training and evaluation logs from Whisper finetuning log file.
Separates training and evaluation metrics into train.csv and eval.csv files.
"""

import json
import csv
from pathlib import Path


def extract_logs(log_file: str, train_csv: str = "train.csv", eval_csv: str = "eval.csv"):
    """
    Extract training and evaluation logs from a finetuning log file.
    
    Args:
        log_file: Path to the input log file
        train_csv: Output path for training metrics CSV
        eval_csv: Output path for evaluation metrics CSV
    """
    train_logs = []
    eval_logs = []
    
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            if not line or not line.startswith('{'):
                continue
            
            try:
                # Convert Python dict format (single quotes) to JSON (double quotes)
                json_line = line.replace("'", '"')
                data = json.loads(json_line)
            except json.JSONDecodeError:
                continue
            
            # Separate training and evaluation logs
            if 'eval_loss' in data:
                eval_logs.append(data)
            elif 'loss' in data:
                train_logs.append(data)
    
    # Write training logs to CSV
    if train_logs:
        train_keys = set()
        for log in train_logs:
            train_keys.update(log.keys())
        train_keys = sorted(list(train_keys))
        
        with open(train_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=train_keys)
            writer.writeheader()
            writer.writerows(train_logs)
        
        print(f"✓ Training logs extracted to {train_csv} ({len(train_logs)} rows)")
    
    # Write evaluation logs to CSV
    if eval_logs:
        eval_keys = set()
        for log in eval_logs:
            eval_keys.update(log.keys())
        eval_keys = sorted(list(eval_keys))
        
        with open(eval_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=eval_keys)
            writer.writeheader()
            writer.writerows(eval_logs)
        
        print(f"✓ Evaluation logs extracted to {eval_csv} ({len(eval_logs)} rows)")


if __name__ == "__main__":
    extract_logs("logs1.txt", train_csv="train1.csv", eval_csv="eval1.csv")
