from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

# Clone model & processor
model_name = "google/paligemma-3b-mix-224"
local_dir = "paligemma-3b-mix-224"

print("Downloading model...")
model = PaliGemmaForConditionalGeneration.from_pretrained(model_name)
model.save_pretrained(local_dir)

print("Downloading processor...")
processor = PaliGemmaProcessor.from_pretrained(model_name)
processor.save_pretrained(local_dir)

print(f"Model & processor saved to: {local_dir}")

