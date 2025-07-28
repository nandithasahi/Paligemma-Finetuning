import os
import sys
import numpy as np
sys.modules["np"] = np

from flask import Flask, jsonify, request
from ModelFinetuningFactory import ModelFinetuningFactory
import logging
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("paligemmaLora")

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train_model():
    try:
        input_data = request.get_json()
        logger.info(f"📥 Input JSON: {input_data}")
        
        # Validate required parameters
        required = ["model_name", "train_data_path", "output_dir"]
        if not all(k in input_data for k in required):
            return jsonify({"status": "error", "message": "Missing required parameters"}), 400

        model_trainer = ModelFinetuningFactory(input_data)

        thread = threading.Thread(target=model_trainer.train)
        thread.start()

        return jsonify({"status": "success", "message": "Training started in background."})
    except Exception as e:
        logger.error(f"❌ Error starting training: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    logger.info("🚀 Starting Flask app...")
    app.run(host="0.0.0.0", port=5000, debug=False)