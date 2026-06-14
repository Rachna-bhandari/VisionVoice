"""
VisionVoice — main.py (Step 4 — Live Demo)
Run after: extract_features.py → preprocess.py → train.py
"""

import os, pickle, numpy as np, matplotlib.pyplot as plt, tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from camera import capture_image
from voice  import speak_caption

DATA_DIR        = "data"
MODEL_PATH      = os.path.join(DATA_DIR, "vision_voice_model.keras")
TOKENIZER_PATH  = os.path.join(DATA_DIR, "tokenizer.pkl")
MAX_LENGTH_PATH = os.path.join(DATA_DIR, "max_length.txt")

def load_resources():
    for path in [MODEL_PATH, TOKENIZER_PATH, MAX_LENGTH_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: '{path}' — run Steps 1-3 first.")
    print("Loading model...")
    try:
        model = load_model(MODEL_PATH)
    except Exception:
        model = load_model(MODEL_PATH, compile=False)
    tokenizer = pickle.load(open(TOKENIZER_PATH, 'rb'))
    with open(MAX_LENGTH_PATH) as f:
        max_length = int(f.read().strip())
    iv3_base       = InceptionV3(weights='imagenet')
    feat_extractor = Model(inputs=iv3_base.input, outputs=iv3_base.layers[-2].output)
    print("✅ Ready!\n")
    return model, tokenizer, max_length, feat_extractor

def predict_caption(image_path, model, tokenizer, max_length, feat_extractor):
    img     = load_img(image_path, target_size=(299, 299))
    img     = img_to_array(img)
    img     = np.expand_dims(img, axis=0)
    img     = preprocess_input(img)
    feature = feat_extractor.predict(img, verbose=0)
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        yhat     = model.predict([feature, sequence], verbose=0)
        yhat     = np.argmax(yhat)
        word     = next((w for w, idx in tokenizer.word_index.items() if idx == yhat), None)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return in_text.replace('startseq', '').strip()

def main():
    model, tokenizer, max_length, feat_extractor = load_resources()
    image_path = capture_image()
    if image_path is None:
        print("No image captured.")
        return
    print("\n🤖 Analysing scene...")
    caption = predict_caption(image_path, model, tokenizer, max_length, feat_extractor)
    print(f"\n📝 Caption: {caption}\n")
    plt.figure(figsize=(8, 6))
    plt.imshow(load_img(image_path))
    plt.title(f"VisionVoice:\n{caption}", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    speak_caption(caption)

if __name__ == "__main__":
    main()
