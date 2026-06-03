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

def _decode(a: Any):
    # Handle dict-like audio (datasets Audio feature), torchcodec AudioDecoder, and metadata objects
    if isinstance(a, dict):
        # common dict: {"array": np.array, "sampling_rate": int}
        if "array" in a:
            return a["array"], a.get("sampling_rate", 16000)
        # fallback: path/bytes - let decode_example handle if needed
        if "path" in a or "bytes" in a:
            # Return as-is; caller can decide to use decode_example on the audio feature
            return a, None
    # torchcodec AudioDecoder objects provide get_all_samples() and metadata attribute
    if hasattr(a, "get_all_samples"):
        arr = a.get_all_samples()
        md = getattr(a, "metadata", None)
        if isinstance(md, dict):
            sr = md.get("sampling_rate", md.get("sample_rate", 16000))
        else:
            sr = getattr(md, "sampling_rate", getattr(md, "sample_rate", 16000))
        return arr, sr
    raise TypeError(f"Unknown audio type: {type(a)}")

def preprocess(batch):
    arrays, srs = [], []
    for a in batch["audio"]:
        arr, sr = _decode(a)
        arrays.append(arr)
        srs.append(sr)
    sr = srs[0] if srs else 16000
    #audios = batch["audio"]
    #arrays = [a["array"] for a in audios]
    inputs = feature_extractor(arrays, sampling_rate=sr, padding=True, return_tensors="np")
 
    batch["input_features"] = inputs["input_features"].tolist()
    batch["labels"] = batch["label"]
 
    return batch

@dataclass
class DataCollator:
    feature_extractor: Any
    def __call__(self, features):
        inputs = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.feature_extractor.pad(inputs, return_tensors="pt")
        batch["labels"] = torch.tensor([feature["labels"] for feature in features], dtype=torch.long)

        return batch

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
    model = WhisperForAudioClassification.from_pretrained("openai/whisper-medium")

    # DATASET
    ds = load_dataset("audiofolder", data_dir="./data_16k/")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    ds = ds.map(preprocess, batched=True, batch_size=32, num_proc=1)

    data_collator = DataCollator(feature_extractor=feature_extractor)

    # PEFT
    randlora_config = RandLoraConfig(
        r=32,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        randlora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, randlora_config)

    # configs for LID
    model.config.num_labels = 8
    model.config.id2label = {0: "ag", 1: "be", 2: "bs", 3: "gr", 4: "lu", 5: "sg", 6: "vs", 7: "zh"}
    model.config.label2id = {"ag": 0, "be": 1, "bs": 2, "gr": 3, "lu": 4, "sg": 5, "vs": 6, "zh": 7}

    # classifier size mismatch patch
    hidden = getattr(model.config, "d_model", getattr(model.config, "hidden_size", None))
    model.classifier = torch.nn.Linear(hidden, model.config.num_labels)

    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="whisper_randlora/exp/",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_steps=500,
        save_steps=500,
        learning_rate=1e-4,
        num_train_epochs=3,
        fp16=True,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
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
