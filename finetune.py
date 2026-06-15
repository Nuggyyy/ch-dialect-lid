from peft import RandLoraConfig, get_peft_model
import torch
import numpy as np
from dataclasses import dataclass
from typing import Any
from datasets import load_dataset, Audio
from sklearn.metrics import accuracy_score
from transformers import (
        AutoFeatureExtractor,
        WhisperForAudioClassification,
        Trainer,
        TrainingArguments,
)

@dataclass
class DataCollator:
    feature_extractor: Any
    label2id: dict = None

    def __call__(self, features):
        # Extract and decode audio samples
        arrays = []
        labels = []

        for feature in features:
            audio_decoder = feature["audio"]
            audio_samples = audio_decoder.get_all_samples()
            # Convert torch tensor to numpy array
            arr = np.asarray(audio_samples.data).squeeze().astype(np.float32)
            arrays.append(arr)
            # Convert string label to numeric ID
            label_str = feature["dialect_region"]
            label_id = self.label2id.get(label_str, 0)
            labels.append(label_id)

        # Get sampling rate from the first sample's metadata
        sr = getattr(audio_decoder.metadata, "sample_rate", 16000)

        # Extract features without padding first
        inputs = self.feature_extractor(
            arrays,
            sampling_rate=sr,
            return_tensors="np"
        )

        # Now pad using the feature extractor's pad method
        inputs = self.feature_extractor.pad(inputs, return_tensors="pt")
        inputs["labels"] = torch.tensor(labels, dtype=torch.long)

        return inputs

def compute_metrics(p):
    preds = p.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    y_pred = np.argmax(preds, axis=1)
    y_true = p.label_ids
    acc = accuracy_score(y_true, y_pred)
    return {"accuracy": acc}

if __name__ == "__main__":
    # MODEL
    feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-medium")
    model = WhisperForAudioClassification.from_pretrained(
        "openai/whisper-medium",
        num_labels=7,
        id2label={0: "Zürich", 1: "Innerschweiz", 2: "Wallis", 3: "Basel", 4: "Graubünden", 5: "Bern", 6: "Ostschweiz"},
        label2id={"Zürich": 0, "Innerschweiz": 1, "Wallis": 2, "Basel": 3, "Graubünden": 4, "Bern": 5, "Ostschweiz": 6},
    )

    # DATASET
    data_files = {
        "train": "./data_16k/train_balanced.tsv",
        "test": "./data_16k/test.tsv",
    }
    ds = load_dataset("csv", data_files=data_files, sep="\t")

    label2id = {"Zürich": 0, "Innerschweiz": 1, "Wallis": 2, "Basel": 3, "Graubünden": 4, "Bern": 5, "Ostschweiz": 6}

    def test_add_audio(batch, dir="./data_16k/clips__test"):
        if dir:
            ids = batch.get("path")
            batch["audio"] = [
                f"{dir}/{fname[:-4]}.wav"  # Remove '.mp3' (4 chars) and add '.wav'
                for fname in ids
            ]
        return batch

    def train_add_audio(batch, dir="./data_16k/clips__train_valid"):
        if dir:
            ids = batch.get("path")
            batch["audio"] = [
                f"{dir}/{fname[:-5]}.wav"  # Remove '.flac' (5 chars) and add '.wav'
                for fname in ids
            ]
        return batch

    ds["train"] = ds["train"].map(train_add_audio, batched=True)
    ds["test"] = ds["test"].map(test_add_audio, batched=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    data_collator = DataCollator(feature_extractor=feature_extractor, label2id=label2id)

    # PEFT
    randlora_config = RandLoraConfig(
        r=32,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        randlora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, randlora_config)

    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="exp/",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        eval_accumulation_steps=4,
        eval_on_start=True,
        eval_steps=1000,
        save_steps=1000,
        eval_strategy="steps",
        save_strategy="best",
        learning_rate=5e-5,
        num_train_epochs=1,
        fp16=True,
        fp16_full_eval=True,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        remove_unused_columns=False,
        optim="adamw_torch_fused"
    )

    trainer = Trainer(
        args=training_args,
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("Saving Model")
    trainer.save_model(training_args.output_dir)
