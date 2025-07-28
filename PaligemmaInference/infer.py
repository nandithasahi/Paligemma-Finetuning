import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel
from PIL import Image
import os

# 🔧 Configuration
BASE_MODEL = "models/paligemma-3b-mix-224"
LORA_DIR = os.path.join("output", "paligemma-finetuned")
IMAGE_PATH = "C:/Users/nandi/myint/PaligemmaInference/data/panda.jpg" 
MAX_NEW_TOKENS = 100

def main():
    # 🔄 Load model and processor
    print("🔄 Loading model and processor...")
    
    try:
        processor = PaliGemmaProcessor.from_pretrained(BASE_MODEL)
        base_model = PaliGemmaForConditionalGeneration.from_pretrained(BASE_MODEL)
        
        # Check if LoRA directory exists
        if not os.path.exists(LORA_DIR):
            print(f"❌ LoRA directory not found: {LORA_DIR}")
            print("Please ensure the model has been fine-tuned first.")
            return
        
        print(f"📂 Loading LoRA weights from {LORA_DIR}...")
        model = PeftModel.from_pretrained(base_model, LORA_DIR)
        
        # 🔀 Merge LoRA weights (important for inference)
        print("🔀 Merging LoRA weights...")
        model = model.merge_and_unload()
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # ✅ Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Using device: {device}")
    model = model.to(device).eval()

    # 🖼️ Load and prepare image
    print(f"📂 Loading image from {IMAGE_PATH}...")
    
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Image file not found: {IMAGE_PATH}")
        return
    
    try:
        image = Image.open(IMAGE_PATH).convert("RGB")
        print(f"✅ Image loaded successfully. Size: {image.size}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return

   
    prompt = "caption en"  
    
    print(f"🔤 Using prompt: '{prompt}'")

    try:
       
        inputs = processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(device)
        
        print("✅ Input processing successful")
        print(f"📊 Input shapes:")
        print(f"  - input_ids: {inputs['input_ids'].shape}")
        print(f"  - attention_mask: {inputs['attention_mask'].shape}")
        print(f"  - pixel_values: {inputs['pixel_values'].shape}")
        
    except Exception as e:
        print(f"❌ Error processing inputs: {e}")
        return

    # 🚀 Generate caption
    print("\n🚀 Running inference...")
    
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                repetition_penalty=1.2,  # Reduced from 3.0
                no_repeat_ngram_size=3
            )
        
        print("✅ Generation completed successfully")
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        return

    # 🔁 Decode output -
    try:
        # Get only the generated tokens (exclude input tokens)
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = generated_ids[:, input_length:]
        
        # Decode the generated tokens
        decoded_output = processor.batch_decode(
            generated_tokens, 
            skip_special_tokens=True
        )[0].strip()
        
        # Alternative: Decode full output and extract meaningful part
        full_decoded = processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0].strip()
        
        print("\n" + "="*50)
        print("🔍 RESULTS:")
        print("="*50)
        print(f"📝 Generated tokens only: '{decoded_output}'")
        print(f"📄 Full decoded output: '{full_decoded}'")
        
        # Try to extract meaningful caption
        if decoded_output and decoded_output.strip():
            print(f"✅ Predicted Caption: {decoded_output}")
        elif full_decoded and full_decoded.strip():
            # Try to extract caption after <image> token
            if "<image>" in full_decoded:
                caption = full_decoded.split("<image>")[-1].strip()
                if caption:
                    print(f"✅ Extracted Caption: {caption}")
                else:
                    print("⚠️ No caption found after <image> token")
            else:
                print(f"✅ Full Caption: {full_decoded}")
        else:
            print("⚠️ No meaningful caption generated")
            
    except Exception as e:
        print(f"❌ Error decoding output: {e}")
        return


if __name__ == "__main__":
    main()