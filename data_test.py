from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor

#ds = DatasetDict()
#ds["train"] = load_dataset("audiofolder", data_dir="/data/train")
#ds["test"] = load_dataset("audiofolder", data_dir="/data/test")

ds = load_dataset("audiofolder", data_dir="./data/")

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
decoded = audio_feat.decode_example(ds["train"][0]["audio"])
print("decoded keys:", decoded.keys())
print("sampling_rate:", decoded["sampling_rate"], "array len:", len(decoded["array"]))

fe = AutoFeatureExtractor.from_pretrained("openai/whisper-medium")
ins = fe([decoded["array"]], sampling_rate=decoded["sampling_rate"], return_tensors="np", padding=True)
print("features shape:", ins["input_features"].shape)
