import os
import torch
import numpy as np
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration, 
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
from PIL import Image
from peft import get_peft_model, LoraConfig, TaskType
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            labels=inputs["labels"]
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

class ModelFinetuningFactory:
    def __init__(self, input_json):
        self.model_name = input_json.get("model_name", "google/paligemma-3b-mix-224")
        self.train_data_path = input_json.get("train_data_path", None)
        self.output_dir = input_json.get("output_dir", "./paligemma-finetuned")
        self.epochs = input_json.get("epochs", 3)
        self.per_device_batch_size = input_json.get("batch_size", 1)
        self.learning_rate = input_json.get("learning_rate", 1e-5)
        self.gradient_accumulation_steps = input_json.get("gradient_accumulation_steps", 16)

    def train(self):
        # Check GPU availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🖥️ Using device: {device}")
        
        print(f"🔄 Loading model and processor from {self.model_name}...")
        processor = PaliGemmaProcessor.from_pretrained(self.model_name)
        model = PaliGemmaForConditionalGeneration.from_pretrained(self.model_name)

        # ✅ Correct LoRA config
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        print(f"📂 Loading dataset from {self.train_data_path}...")
        dataset = load_dataset("json", data_files=self.train_data_path, split="train")

        def transform_example(example):
            # ✅ Fixed path handling
            image_path = os.path.join(os.path.dirname(self.train_data_path), example["image_path"])
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
                
            # ✅ Load image without normalization
            image = Image.open(image_path).convert("RGB")
            
            prompt = "caption en"
            full_text = f"<image>{prompt}\n{example['caption']}"

            # ✅ Let processor handle normalization
            encoding = processor(
                images=image,
                text=full_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=256
            )

            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
            pixel_values = encoding["pixel_values"].squeeze(0)
            labels = input_ids.clone()

            # ✅ Mask prompt tokens
            prompt_length = processor.tokenizer.encode(f"<image>{prompt}\n", return_tensors="pt").shape[1]
            labels[:prompt_length] = -100

            return {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }

        print("🔄 Preprocessing dataset...")
        dataset = dataset.map(transform_example, batched=False, remove_columns=dataset.column_names)
        dataset.set_format(type="torch")
        
        # Move dataset to appropriate device
        dataset = dataset.with_format("torch", device=device)

        # Print sample for verification
        sample = dataset[0]
        print(f"📊 Sample input shape: {sample['input_ids'].shape}")
        print(f"📊 Sample pixel values shape: {sample['pixel_values'].shape}")

        # ✅ CPU-optimized TrainingArguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.per_device_batch_size,
            num_train_epochs=self.epochs,
            learning_rate=self.learning_rate,
            warmup_steps=5,
            logging_steps=5,
            save_steps=10,
            save_total_limit=1,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            dataloader_pin_memory=False,
            fp16=False,
            logging_dir=os.path.join(self.output_dir, "logs"),
            report_to="none",
            remove_unused_columns=False,
            load_best_model_at_end=False,
            dataloader_num_workers=0,
            per_device_eval_batch_size=1,
            lr_scheduler_type="linear",
            max_grad_norm=0.5,
            disable_tqdm=False,
        )

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=lambda data: {
                'pixel_values': torch.stack([d['pixel_values'] for d in data]),
                'input_ids': torch.stack([d['input_ids'] for d in data]),
                'attention_mask': torch.stack([d['attention_mask'] for d in data]),
                'labels': torch.stack([d['labels'] for d in data])
            }
        )

        print("🚀 Starting LoRA fine-tuning...")
        trainer.train()

        print(f"💾 Saving LoRA fine-tuned model to {self.output_dir}...")
        trainer.save_model(self.output_dir)
        processor.save_pretrained(self.output_dir)

        return {
            "message": "✅ LoRA Training completed with PaliGemma format.",
            "output_dir": self.output_dir
        }

if __name__ == "__main__":
    import json
    config = {
        "model_name": r"C:\Users\nandi\myint\PaligemmaInference\models\paligemma-3b-mix-224",
        "train_data_path": r"C:\Users\nandi\myint\PaligemmaInference\data\train.jsonl",
        "output_dir": r"C:\Users\nandi\myint\PaligemmaInference\output",
        "epochs": 3,
        "batch_size": 1,
        "learning_rate": 1e-5,
        "gradient_accumulation_steps": 16
    }
    
    factory = ModelFinetuningFactory(config)
    result = factory.train()
    print(result)