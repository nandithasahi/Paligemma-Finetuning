import requests
import os
import json

url = "http://127.0.0.1:5000/train"

# ✅ Fix 1: Escape backslashes in Windows paths or use raw strings
config = {
    "model_name": r"C:\Users\nandi\myint\PaligemmaInference\models\paligemma-3b-mix-224",
    "train_data_path": r"C:\Users\nandi\myint\PaligemmaInference\data\train.jsonl",
    "output_dir": r"C:\Users\nandi\myint\PaligemmaInference\output",
    "epochs": 3,
    "batch_size": 1,
    "learning_rate": 1e-4,
    "gradient_accumulation_steps": 4
}

print("📡 Sending training request...")
try:
    # ✅ Fix 2: Add timeout and error handling
    response = requests.post(url, json=config, timeout=30)
    response.raise_for_status()  # Raises an exception for HTTP errors
    
    print(f"✅ Status Code: {response.status_code}")
    print(f"📦 Response: {response.json()}")
    
except requests.exceptions.RequestException as e:
    print(f"❌ Error sending request: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"Response content: {e.response.text}")
except json.JSONDecodeError as e:
    print(f"❌ Error decoding JSON response: {e}")
    print(f"Raw response: {response.text}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")