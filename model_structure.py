"""
VisionVoice — model_structure.py
Prints the saved model architecture.
"""

import os, pickle
from tensorflow.keras.models import load_model

DATA_DIR        = "data"
CHECKPOINT_PATH = os.path.join(DATA_DIR, "vision_voice_model.keras")

def print_model_summary():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[ERROR] Model not found. Run train.py first.")
        return
    model = load_model(CHECKPOINT_PATH, compile=False)
    model.summary()
    tokenizer_path = os.path.join(DATA_DIR, 'tokenizer.pkl')
    if os.path.exists(tokenizer_path):
        tokenizer = pickle.load(open(tokenizer_path, 'rb'))
        print(f"\nVocabulary size : {len(tokenizer.word_index)+1}")
    max_len_path = os.path.join(DATA_DIR, 'max_length.txt')
    if os.path.exists(max_len_path):
        with open(max_len_path) as f:
            print(f"Max caption len : {f.read().strip()}")

if __name__ == "__main__":
    print_model_summary()
