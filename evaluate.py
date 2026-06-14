"""
VisionVoice — evaluate.py
Evaluates the trained model using BLEU-1/2/3/4 scores on a sample of images.
Saves a validation chart to data/validation_results.png
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import nltk
nltk.download('punkt', quiet=True)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
MODEL_PATH = os.path.join(DATA_DIR, "vision_voice_model.keras")

# ── Load resources ────────────────────────────────────────────────────────────
print("Loading resources …")
model     = load_model(MODEL_PATH, compile=False)
tokenizer = pickle.load(open(os.path.join(DATA_DIR, 'tokenizer.pkl'), 'rb'))
mapping   = pickle.load(open(os.path.join(DATA_DIR, 'mapping.pkl'),   'rb'))
features  = pickle.load(open(os.path.join(DATA_DIR, 'features.pkl'),  'rb'))

with open(os.path.join(DATA_DIR, 'max_length.txt'), 'r') as f:
    max_length = int(f.read().strip())

print(f"Vocab size : {len(tokenizer.word_index) + 1}")
print(f"Max length : {max_length}")
print(f"Images     : {len(mapping)}\n")


# ── Caption generator ─────────────────────────────────────────────────────────
def generate_caption(feature):
    in_text = 'startseq'
    for _ in range(max_length):
        seq  = tokenizer.texts_to_sequences([in_text])[0]
        seq  = pad_sequences([seq], maxlen=max_length, padding='post')
        yhat = model.predict([feature.reshape(1, -1), seq], verbose=0)
        yhat = np.argmax(yhat)
        word = next(
            (w for w, idx in tokenizer.word_index.items() if idx == yhat),
            None
        )
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return in_text.replace('startseq', '').strip()


# ── Evaluation loop ────────────────────────────────────────────────────────────
EVAL_SAMPLE  = 200
common_keys  = [k for k in mapping.keys() if k in features]
eval_keys    = common_keys[:EVAL_SAMPLE]
smoothing    = SmoothingFunction().method1

actual_corpus    = []
predicted_corpus = []
bleu1_scores     = []
bleu2_scores     = []

print(f"Evaluating on {len(eval_keys)} images …\n")

for i, key in enumerate(eval_keys):
    feature   = features[key]
    predicted = generate_caption(feature)

    references = [
        cap.replace('startseq', '').replace('endseq', '').strip().split()
        for cap in mapping[key]
    ]
    predicted_tokens = predicted.split()

    actual_corpus.append(references)
    predicted_corpus.append(predicted_tokens)

    b1 = corpus_bleu([references], [predicted_tokens], weights=(1, 0, 0, 0))
    b2 = corpus_bleu([references], [predicted_tokens], weights=(0.5, 0.5, 0, 0))
    bleu1_scores.append(b1)
    bleu2_scores.append(b2)

    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(eval_keys)} done …")

# ── Final BLEU scores ─────────────────────────────────────────────────────────
bleu1 = corpus_bleu(actual_corpus, predicted_corpus, weights=(1.0,  0,    0,    0   ))
bleu2 = corpus_bleu(actual_corpus, predicted_corpus, weights=(0.5,  0.5,  0,    0   ))
bleu3 = corpus_bleu(actual_corpus, predicted_corpus, weights=(0.33, 0.33, 0.33, 0   ))
bleu4 = corpus_bleu(actual_corpus, predicted_corpus, weights=(0.25, 0.25, 0.25, 0.25))

print("\n" + "="*45)
print("  VisionVoice — Validation Report")
print("="*45)
print(f"  Images evaluated : {len(eval_keys)}")
print(f"  BLEU-1 : {bleu1*100:.2f}%  (word accuracy)")
print(f"  BLEU-2 : {bleu2*100:.2f}%  (2-word phrases)")
print(f"  BLEU-3 : {bleu3*100:.2f}%  (3-word phrases)")
print(f"  BLEU-4 : {bleu4*100:.2f}%  (sentence fluency)")
print(f"  Avg BLEU-1/image : {np.mean(bleu1_scores)*100:.2f}%")
print(f"  Avg BLEU-2/image : {np.mean(bleu2_scores)*100:.2f}%")
print("="*45)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('VisionVoice — Validation Accuracy', fontsize=14)

scores = [bleu1*100, bleu2*100, bleu3*100, bleu4*100]
labels = ['BLEU-1\n(words)', 'BLEU-2\n(2-gram)', 'BLEU-3\n(3-gram)', 'BLEU-4\n(fluency)']
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']

bars = axes[0].bar(labels, scores, color=colors, width=0.5)
axes[0].set_title('BLEU Score Breakdown')
axes[0].set_ylabel('Score (%)')
axes[0].set_ylim(0, 100)
axes[0].axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
for bar, score in zip(bars, scores):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1,
                 f'{score:.1f}%', ha='center', fontsize=11, fontweight='bold')
axes[0].legend()

axes[1].hist(np.array(bleu1_scores) * 100, bins=20,
             color='#3498db', edgecolor='white', alpha=0.8)
axes[1].axvline(x=np.mean(bleu1_scores) * 100, color='red', linestyle='--',
                label=f'Mean: {np.mean(bleu1_scores)*100:.1f}%')
axes[1].set_title('BLEU-1 Score Distribution per Image')
axes[1].set_xlabel('BLEU-1 Score (%)')
axes[1].set_ylabel('Number of Images')
axes[1].legend()

plt.tight_layout()
out_path = os.path.join(DATA_DIR, 'validation_results.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\n✅ Chart saved → {out_path}")

# ── Sample predictions ────────────────────────────────────────────────────────
print("\n" + "="*45)
print("  Sample Predictions")
print("="*45)
for i, key in enumerate(eval_keys[:5], 1):
    predicted = generate_caption(features[key])
    actual    = mapping[key][0].replace('startseq', '').replace('endseq', '').strip()
    print(f"\n[{i}] Image     : {key}")
    print(f"    Predicted : {predicted}")
    print(f"    Actual    : {actual}")
