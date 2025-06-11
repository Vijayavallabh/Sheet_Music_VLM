

import torch
import os
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
from PIL import Image

class InferenceHandler:
    def __init__(self, final_adapter_path, config):
        print("\n--- Preparing final model for Inference ---")
        if os.path.exists(final_adapter_path):
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=final_adapter_path, max_seq_length=config.MAX_SEQ_LENGTH,
                dtype=None, load_in_4bit=True,
            )
            print("Successfully loaded fine-tuned adapters with Unsloth.")
        else:
            print("WARNING: Final adapter not found. Loading base model for demo.")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=config.MODEL_ID, max_seq_length=config.MAX_SEQ_LENGTH,
                dtype=None, load_in_4bit=True,
            )
        self.image_processor = self.model.vision_language_model.image_processor

    def predict(self, image, text_prompt):
        if image is None or not text_prompt.strip():
            return "Please provide both an image and a text prompt."

        messages = [{"role": "user", "content": f"<image>\n{text_prompt}"}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        pixel_values = self.image_processor(images=[image], return_tensors="pt")["pixel_values"].to("cuda")
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = self.model.generate(
            input_ids=inputs["input_ids"], pixel_values=pixel_values,
            attention_mask=inputs["attention_mask"], max_new_tokens=512, use_cache=True
        )
        response_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        assistant_response = response_text.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1].strip()
        return assistant_response