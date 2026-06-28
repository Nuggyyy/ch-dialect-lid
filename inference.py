import torch
import numpy as np
from dataclasses import dataclass
from typing import Any
from datasets import load_dataset, Audio
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from transformers import (
    AutoFeatureExtractor,
    WhisperForAudioClassification,
)
from peft import PeftModel
from peft import PeftModel


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


def run_inference(model_path="./exp/", batch_size=8):
    """
    Load fine-tuned model and run inference on test set.
    Compute macro F1 score and other metrics.
    """
    # Define labels and mappings
    label2id = {
        "Zürich": 0,
        "Innerschweiz": 1,
        "Wallis": 2,
        "Basel": 3,
        "Graubünden": 4,
        "Bern": 5,
        "Ostschweiz": 6,
    }
    id2label = {v: k for k, v in label2id.items()}

    # Load feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-medium")

    # Load base model first
    print(f"Loading base model...")
    from peft import PeftModel
    base_model = WhisperForAudioClassification.from_pretrained(
        "openai/whisper-medium",
        num_labels=7,
        id2label=id2label,
        label2id=label2id,
    )

    # Load PEFT adapter
    print(f"Loading PEFT adapter from {model_path}...")
    model = PeftModel.from_pretrained(base_model, model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"Model loaded on device: {device}")

    # Load test dataset
    print("Loading test dataset...")
    data_files = {"test": "./data_16k/test.tsv"}
    ds = load_dataset("csv", data_files=data_files, sep="\t")

    def test_add_audio(batch, dir="./data_16k/clips__test"):
        if dir:
            ids = batch.get("path")
            batch["audio"] = [
                f"{dir}/{fname[:-4]}.wav"  # Remove '.mp3' (4 chars) and add '.wav'
                for fname in ids
            ]
        return batch

    ds["test"] = ds["test"].map(test_add_audio, batched=True)
    ds["test"] = ds["test"].cast_column("audio", Audio(sampling_rate=16000))

    data_collator = DataCollator(feature_extractor=feature_extractor, label2id=label2id)

    # Run inference
    print("Running inference...")
    all_preds = []
    all_labels = []

    # Process in batches
    for i in range(0, len(ds["test"]), batch_size):
        batch_data = ds["test"][i : i + batch_size]
        # Convert sliced dataset (dict of lists) to list of dicts
        features = [
            {k: v[j] for k, v in batch_data.items()}
            for j in range(len(batch_data["audio"]))
        ]
        batch = data_collator(features)

        with torch.no_grad():
            outputs = model(
                input_features=batch["input_features"].to(device),
                return_dict=True,
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print("\n" + "=" * 60)
    print("INFERENCE RESULTS")
    print("=" * 60)
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Macro F1:     {macro_f1:.4f}")
    print(f"Weighted F1:  {weighted_f1:.4f}")
    print("=" * 60)

    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=[id2label[i] for i in range(len(id2label))],
        zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(id2label))))
    print("\nConfusion Matrix (rows=true labels, columns=predicted labels):")
    labels_names = [id2label[i] for i in range(len(id2label))]
    header = "".join(f"{name[:10]:>10}" for name in labels_names)
    print(f"{'':20}{header}")
    for i, row in enumerate(cm):
        row_str = "".join(f"{int(x):10d}" for x in row)
        print(f"{labels_names[i]:<20}{row_str}")

    # Normalized confusion matrix (by true labels)
    with np.errstate(all='ignore'):
        cm_norm = cm.astype('float')
        row_sums = cm_norm.sum(axis=1)
        # avoid division by zero
        row_sums[row_sums == 0] = 1
        cm_norm = cm_norm / row_sums[:, np.newaxis]

    print("\nNormalized Confusion Matrix (rows normalized to sum to 1):")
    header = "".join(f"{name[:10]:>10}" for name in labels_names)
    print(f"{'':20}{header}")
    for i, row in enumerate(cm_norm):
        row_str = "".join(f"{x:10.3f}" for x in row)
        print(f"{labels_names[i]:<20}{row_str}")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "predictions": all_preds,
        "labels": all_labels,
    }


if __name__ == "__main__":
    results = run_inference()
