from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor

#ds = DatasetDict()
#ds["train"] = load_dataset("audiofolder", data_dir="/data/train")
#ds["test"] = load_dataset("audiofolder", data_dir="/data/test")

data_files = {
    "train": "./data_16k/train_balanced.tsv",
    "test": "./data_16k/test.tsv",
}
ds = load_dataset("csv", data_files=data_files, sep="\t")

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

####

print(ds)
print(ds["train"].num_rows)
print(ds["train"][0])
print(ds["test"].num_rows)
print(ds["test"][0])

a = ds["train"][0]["audio"]
print("type:", type(a))
print("repr:", repr(a))
print("dir sample:", [n for n in dir(a) if not n.startswith("_")][:40])
#print(type(a), a.get("sampling_rate"), len(a.get("array")))

####

audio_feat = ds["train"].features["audio"]

# Use with_format to get properly decoded audio
ds_formatted = ds.with_format("numpy")
audio_data = ds_formatted["train"][0]["audio"]

print("audio dict keys:", audio_data.keys())
print("sampling_rate:", audio_data["sampling_rate"], "array len:", len(audio_data["array"]))

fe = AutoFeatureExtractor.from_pretrained("openai/whisper-medium")
ins = fe([audio_data["array"]], sampling_rate=audio_data["sampling_rate"], return_tensors="np", padding=True)
print("features shape:", ins["input_features"].shape)

