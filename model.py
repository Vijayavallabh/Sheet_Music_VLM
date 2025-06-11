import torch
import os
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
class ModelTrainer:
    def __init__(self, config):
        self.config = config

    def train_phase(self, phase_num, train_dataset, output_dir, input_adapter_path=None):
        print(f"\n--- Starting Training Phase {phase_num}: Outputting to {output_dir} ---")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.MODEL_ID, max_seq_length=self.config.MAX_SEQ_LENGTH,
            dtype=None, load_in_4bit=True
        )
        if input_adapter_path and os.path.exists(input_adapter_path):
            print(f"Loading adapters from previous phase: {input_adapter_path}")
            model = FastLanguageModel.get_peft_model(model, peft_model_id=input_adapter_path)
        else:
            model = FastLanguageModel.get_peft_model(
                model, r=self.config.LORA_R, target_modules=self.config.LORA_TARGET_MODULES,
                lora_alpha=self.config.LORA_ALPHA, lora_dropout=self.config.LORA_DROPOUT,
                bias="none", use_gradient_checkpointing=True, random_state=42
            )
        
        training_args = TrainingArguments(
            output_dir=output_dir, per_device_train_batch_size=2,
            gradient_accumulation_steps=4, warmup_steps=5, num_train_epochs=1,
            learning_rate=2e-4 if phase_num == 1 else 1e-4,
            bf16=torch.cuda.is_bfloat16_supported(), logging_steps=5,
            optim="adamw_8bit", weight_decay=0.01, seed=42,
        )

        trainer = SFTTrainer(
            model=model, tokenizer=tokenizer, train_dataset=train_dataset,
            dataset_text_field="text", dataset_image_field="images",
            max_seq_length=self.config.MAX_SEQ_LENGTH, args=training_args,
        )
        
        print(f"Starting actual training for Phase {phase_num}...")
        trainer.train()
        print(f"--- PHASE {phase_num} TRAINING COMPLETE (SIMULATED) ---")