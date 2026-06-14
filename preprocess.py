"""
VisionVoice — Step 2: Caption Preprocessing & Tokenization
Reads data/captions.txt, cleans captions, builds tokenizer.
Outputs: data/tokenizer.pkl, data/mapping.pkl, data/max_length.txt
"""

import os
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

CAPTIONS_PATH = os.path.join("data", "captions.txt")
SAVE_DIR      = "data"


def clean_text(captions_path=CAPTIONS_PATH, save_dir=SAVE_DIR):
    if not os.path.exists(captions_path):
        print(f"[ERROR] '{captions_path}' not found.")
        print("  Download Flickr8k from Kaggle and place captions.txt in data/")
        return

    mapping = {}
    with open(captions_path, 'r', encoding='utf-8') as f:
        next(f)                         # skip header line
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',', 1)
            if len(parts) < 2:
                continue
            img_id  = parts[0].split('.')[0].split('#')[0]
            caption = parts[1].lower().strip()
            mapping.setdefault(img_id, []).append(f"startseq {caption} endseq")

    if not mapping:
        print("[ERROR] No captions parsed. Check the format of captions.txt")
        return

    all_captions = [cap for caps in mapping.values() for cap in caps]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)

    max_length = max(len(cap.split()) for cap in all_captions)

    os.makedirs(save_dir, exist_ok=True)
    pickle.dump(tokenizer, open(os.path.join(save_dir, 'tokenizer.pkl'), 'wb'))
    pickle.dump(mapping,   open(os.path.join(save_dir, 'mapping.pkl'),   'wb'))
    with open(os.path.join(save_dir, 'max_length.txt'), 'w') as f:
        f.write(str(max_length))

    vocab_size = len(tokenizer.word_index) + 1
    print(f"✅ tokenizer.pkl  — vocab = {vocab_size} words")
    print(f"✅ mapping.pkl    — {len(mapping)} images")
    print(f"✅ max_length.txt — {max_length}")


if __name__ == "__main__":
    clean_text()
