import torch
import os
from datasets import load_dataset
from config import Config
class DataManager:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _format_prompt_alignment(self, example):
        prompt_text = "Transcribe the musical notation in this image into ABC format."
        messages = [
            {"role": "user", "content": f"<image>\n{prompt_text}"},
            {"role": "assistant", "content": example['transcription']}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text, "images": example["image"]}

    def _format_prompt_vqa(self, example):
        messages = [
            {"role": "user", "content": example['prompt']},
            {"role": "assistant", "content": example['answer']}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text, "images": example["image"]}

    def get_dataset_for_phase(self, phase_name, sample_size):
        print(f"Loading and preparing dataset for '{phase_name}'...")
        dataset = load_dataset(Config.DATASET_ID, split=f"{phase_name}[:{sample_size}]")
            
        if phase_name == "train_alignment":
            formatted_dataset = dataset.map(self._format_prompt_alignment)
        else:
            formatted_dataset = dataset.map(self._format_prompt_vqa)
            
        print(f"Successfully prepared {len(formatted_dataset)} samples for '{phase_name}'.")
        return formatted_dataset
