from peft import RandLoraConfig, get_peft_model
import torch
from transformers import AutoFeatureExtractor, WhisperForAudioClassification

feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-medium")
model = WhisperForAudioClassification.from_pretrained("openai/whisper-medium")

randlora_config = RandLoraConfig(
    r=32,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
    randlora_dropout=0.1,
    bias="none",
)
model = get_peft_model(model, randlora_config)
model.print_trainable_parameters()
