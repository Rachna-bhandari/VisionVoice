"""
VisionVoice — Step 1: Feature Extraction
Extracts 2048-dim InceptionV3 features from all images in data/Images/
Output: data/features.pkl
"""

import os
import pickle
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

IMAGE_DIR   = os.path.join("data", "Images")
OUTPUT_PATH = os.path.join("data", "features.pkl")
IMAGE_EXTS  = {'.jpg', '.jpeg', '.png', '.bmp'}

# ── Build feature extractor (penultimate layer of InceptionV3) ──────────────
base_model       = InceptionV3(weights='imagenet')
feature_extractor = Model(inputs=base_model.input,
                          outputs=base_model.layers[-2].output)

def extract_features(directory=IMAGE_DIR, output_path=OUTPUT_PATH):
    if not os.path.exists(directory):
        print(f"[ERROR] Directory '{directory}' not found.")
        print("  Make sure you have data/Images/ with Flickr8k images.")
        return

    image_files = [
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ]

    if not image_files:
        print(f"[ERROR] No image files found in '{directory}'.")
        return

    print(f"\nFound {len(image_files)} images. Extracting features...\n")
    features = {}

    for img_name in tqdm(image_files, desc="Extracting"):
        img_path = os.path.join(directory, img_name)
        try:
            image   = load_img(img_path, target_size=(299, 299))
            image   = img_to_array(image)
            image   = np.expand_dims(image, axis=0)
            image   = preprocess_input(image)
            feature = feature_extractor.predict(image, verbose=0)
            img_id  = os.path.splitext(img_name)[0]
            features[img_id] = feature[0]
        except Exception as e:
            print(f"\n[SKIP] '{img_name}': {e}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pickle.dump(features, open(output_path, 'wb'))
    print(f"\n✅ {len(features)} feature vectors saved → '{output_path}'")


if __name__ == "__main__":
    extract_features()
