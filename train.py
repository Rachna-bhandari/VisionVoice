"""
VisionVoice — Step 3: Model Training
Trains an InceptionV3 + LSTM image captioning model.
Resumes automatically from checkpoint if it exists.
Output: data/vision_voice_model.keras
"""

import gc
import os
import random
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Embedding, Input, LSTM, add
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR        = "data"
CHECKPOINT_PATH = os.path.join(DATA_DIR, "vision_voice_model.keras")

# ── Load preprocessed data ───────────────────────────────────────────────────
features  = pickle.load(open(os.path.join(DATA_DIR, 'features.pkl'),  'rb'))
tokenizer = pickle.load(open(os.path.join(DATA_DIR, 'tokenizer.pkl'), 'rb'))
mapping   = pickle.load(open(os.path.join(DATA_DIR, 'mapping.pkl'),   'rb'))

with open(os.path.join(DATA_DIR, 'max_length.txt'), 'r') as f:
    max_length = int(f.read().strip())

vocab_size = len(tokenizer.word_index) + 1

print(f"Vocab size : {vocab_size}")
print(f"Max length : {max_length}")
print(f"Images     : {len(mapping)}")
print(f"Features   : {len(features)}")


# ── Data generator ───────────────────────────────────────────────────────────
def data_generator(mapping, features, tokenizer, max_length, vocab_size, batch_size):
    keys = list(mapping.keys())
    while True:
        random.shuffle(keys)
        for i in range(0, len(keys), batch_size):
            X1, X2, y = [], [], []
            for key in keys[i:i + batch_size]:
                if key not in features:
                    continue
                feature = features[key].astype(np.float32)
                for caption in mapping[key]:
                    seq = tokenizer.texts_to_sequences([caption])[0]
                    for j in range(1, len(seq)):
                        in_seq = pad_sequences(
                            [seq[:j]], maxlen=max_length, padding='post'
                        )[0].astype(np.int32)
                        X1.append(feature)
                        X2.append(in_seq)
                        y.append(seq[j])
            if not X1:
                continue
            yield (
                np.array(X1, dtype=np.float32),
                np.array(X2, dtype=np.int32)
            ), np.array(y, dtype=np.int32)
            del X1, X2, y
            gc.collect()


# ── Build or resume model ────────────────────────────────────────────────────
if os.path.exists(CHECKPOINT_PATH):
    print(f"\nResuming from checkpoint: {CHECKPOINT_PATH}")
    model = load_model(CHECKPOINT_PATH)
else:
    print("\nBuilding new model...")
    inputs1  = Input(shape=(2048,),      name='image_input')
    fe1      = Dropout(0.5)(inputs1)
    fe2      = Dense(512, activation='relu', name='image_dense')(fe1)

    inputs2  = Input(shape=(max_length,), name='sequence_input')
    se1      = Embedding(vocab_size, 512, mask_zero=False, name='embedding')(inputs2)
    se2      = Dropout(0.5)(se1)
    se3      = LSTM(512, name='lstm', use_cudnn=False)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(512, activation='relu')(decoder1)
    outputs  = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

model.summary()

# ── Callbacks ────────────────────────────────────────────────────────────────
callbacks = [
    ModelCheckpoint(
        CHECKPOINT_PATH, monitor='loss',
        save_best_only=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor='loss', factor=0.5,
        patience=3, min_lr=1e-7, verbose=1
    ),
    EarlyStopping(
        monitor='loss', patience=6,
        restore_best_weights=True, verbose=1
    ),
]

# ── Train ────────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS     = 30
steps      = max(1, len(mapping) // BATCH_SIZE)

print(f"\n Vocab      : {vocab_size} words")
print(f" Neurons    : 512")
print(f" Epochs     : {EPOCHS}")
print(f" Batch size : {BATCH_SIZE}")
print(f" Steps/epoch: {steps}")
print(f" Target loss: below 3.0\n")

model.fit(
    data_generator(mapping, features, tokenizer, max_length, vocab_size, BATCH_SIZE),
    steps_per_epoch=steps,
    epochs=EPOCHS,
    verbose=1,
    callbacks=callbacks,
)

print(f"\n✅ Model saved → {CHECKPOINT_PATH}")
