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
            labels.append(feature["label"])

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
        num_labels=8,
        id2label={0: "ag", 1: "be", 2: "bs", 3: "gr", 4: "lu", 5: "sg", 6: "vs", 7: "zh"},
        label2id={"ag": 0, "be": 1, "bs": 2, "gr": 3, "lu": 4, "sg": 5, "vs": 6, "zh": 7},
    )

    # DATASET
    ds = load_dataset("audiofolder", data_dir="./data_16k/")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    data_collator = DataCollator(feature_extractor=feature_extractor)

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
        output_dir="whisper_randlora/exp/",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_steps=500,
        save_steps=500,
        save_strategy="best",
        learning_rate=1e-4,
        num_train_epochs=3,
        fp16=True,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        remove_unused_columns=False,
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
