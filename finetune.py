"""
Finetune WhisperForAudioClassification for Swiss-German dialect ID using audio folder structure.
Audio directory is expected to contain one subfolder per dialect (e.g., data/ch_gr/, data/ch_sg/, ...)
Each audio file can also be named <label>_XXXX.wav; the script prefers the parent folder name as label.

Example usage:
  python finetune.py --audio_dir data\wav --output_dir outputs --per_device_train_batch_size 8 --num_train_epochs 3
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import torch
from datasets import Dataset, Audio
from peft import RandLoraConfig, get_peft_model
from sklearn.metrics import accuracy_score
from transformers import (
    AutoFeatureExtractor,
    WhisperForAudioClassification,
    Trainer,
    TrainingArguments,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--audio_dir", required=True, help="Root audio directory containing one subdir per class")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--model_name", default="openai/whisper-medium")
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sampling_rate", type=int, default=16000)
    return p.parse_args()


class DataCollatorPad:
    """Pad input_features (list of 2D arrays) to the longest in batch."""

    def __call__(self, features):
        input_feats = [f["input_features"] for f in features]
        labels = torch.tensor([f["label"] for f in features], dtype=torch.long)
        seq_lens = [arr.shape[0] for arr in input_feats]
        feat_dim = input_feats[0].shape[1]
        max_len = max(seq_lens)
        batch_size = len(input_feats)
        padded = np.zeros((batch_size, max_len, feat_dim), dtype=np.float32)
        attention_mask = np.zeros((batch_size, max_len), dtype=np.float32)
        for i, arr in enumerate(input_feats):
            l = arr.shape[0]
            padded[i, :l, :] = arr
            attention_mask[i, :l] = 1.0
        batch = {
            "input_features": torch.tensor(padded, dtype=torch.float32),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
            "labels": labels,
        }
        return batch


def build_examples_from_folders(audio_root, allowed_exts={".wav", ".flac", ".mp3"}):
    examples = []
    labels = set()
    for root, dirs, files in os.walk(audio_root):
        # skip the top-level root files if any
        if root == audio_root:
            continue
        parent = os.path.basename(root)
        for f in files:
            name, ext = os.path.splitext(f)
            if ext.lower() in allowed_exts:
                path = os.path.join(root, f)
                label = parent
                # fallback: if parent is not informative, try prefix before underscore
                if not label or label == "":
                    label = name.split("_")[0]
                examples.append({"audio": {"path": path}, "label_name": label})
                labels.add(label)
    labels = sorted(labels)
    label2id = {lab: i for i, lab in enumerate(labels)}
    # convert label_name to id
    for ex in examples:
        ex["label"] = label2id[ex.pop("label_name")]
    return examples, label2id


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading feature extractor and model")
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    model = WhisperForAudioClassification.from_pretrained(args.model_name)

    # apply PEFT RandLora (keeps small number of params trainable)
    randlora_config = RandLoraConfig(
        r=32,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        randlora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, randlora_config)
    model.print_trainable_parameters()

    print("Building dataset from audio folders")
    examples, label2id = build_examples_from_folders(args.audio_dir)
    if not examples:
        raise ValueError("No audio examples found under: %s" % args.audio_dir)
    print(f"Found {len(examples)} audio files across {len(label2id)} labels")

    ds = Dataset.from_list(examples)
    ds = ds.cast_column("audio", Audio(sampling_rate=args.sampling_rate))

    ds = ds.train_test_split(test_size=0.05, seed=args.seed)
    train_ds = ds["train"]
    eval_ds = ds["test"]

    def preprocess(batch):
        audios = [a["array"] for a in batch["audio"]]
        inputs = feature_extractor(audios, sampling_rate=args.sampling_rate)
        batch["input_features"] = inputs["input_features"]
        return batch

    print("Extracting features (this may take time)")
    train_ds = train_ds.map(preprocess, remove_columns=["audio"], batched=True, batch_size=16)
    eval_ds = eval_ds.map(preprocess, remove_columns=["audio"], batched=True, batch_size=16)

    id2label = {v: k for k, v in label2id.items()}
    model.config.id2label = id2label
    model.config.label2id = label2id

    def compute_metrics(p):
        preds = p.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        y_pred = np.argmax(preds, axis=1)
        y_true = p.label_ids
        acc = accuracy_score(y_true, y_pred)
        return {"accuracy": acc}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
    )

    data_collator = DataCollatorPad()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("Saving model")
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
