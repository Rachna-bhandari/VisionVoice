"""
VisionVoice — model_structure.py
Loads the saved model and prints its architecture summary.
Run this anytime to inspect the model without training.
"""

import os
import pickle
from tensorflow.keras.models import load_model

DATA_DIR        = "data"
CHECKPOINT_PATH = os.path.join(DATA_DIR, "vision_voice_model.keras")


def print_model_summary():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[ERROR] Model not found at '{CHECKPOINT_PATH}'")
        print("  Run train.py first to generate the model.")
        return

    print(f"Loading model from: {CHECKPOINT_PATH}\n")
    model = load_model(CHECKPOINT_PATH, compile=False)
    model.summary()

    # Also print vocab / sequence info if available
    tokenizer_path = os.path.join(DATA_DIR, 'tokenizer.pkl')
    max_len_path   = os.path.join(DATA_DIR, 'max_length.txt')

    if os.path.exists(tokenizer_path):
        tokenizer  = pickle.load(open(tokenizer_path, 'rb'))
        vocab_size = len(tokenizer.word_index) + 1
        print(f"\nVocabulary size : {vocab_size}")

    if os.path.exists(max_len_path):
        with open(max_len_path) as f:
            print(f"Max caption len : {f.read().strip()}")


if __name__ == "__main__":
    print_model_summary()
