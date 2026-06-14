"""
VisionVoice — evaluate.py
BLEU-1/2/3/4 evaluation + saves chart to data/validation_results.png
"""

import os, pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import nltk
nltk.download('punkt', quiet=True)

DATA_DIR   = "data"
MODEL_PATH = os.path.join(DATA_DIR, "vision_voice_model.keras")

model     = load_model(MODEL_PATH, compile=False)
tokenizer = pickle.load(open(os.path.join(DATA_DIR, 'tokenizer.pkl'), 'rb'))
mapping   = pickle.load(open(os.path.join(DATA_DIR, 'mapping.pkl'),   'rb'))
features  = pickle.load(open(os.path.join(DATA_DIR, 'features.pkl'),  'rb'))
with open(os.path.join(DATA_DIR, 'max_length.txt')) as f:
    max_length = int(f.read().strip())

def generate_caption(feature):
    in_text = 'startseq'
    for _ in range(max_length):
        seq  = tokenizer.texts_to_sequences([in_text])[0]
        seq  = pad_sequences([seq], maxlen=max_length, padding='post')
        yhat = model.predict([feature.reshape(1,-1), seq], verbose=0)
        yhat = np.argmax(yhat)
        word = next((w for w, idx in tokenizer.word_index.items() if idx == yhat), None)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return in_text.replace('startseq', '').strip()

EVAL_SAMPLE     = 200
common_keys     = [k for k in mapping.keys() if k in features]
eval_keys       = common_keys[:EVAL_SAMPLE]
actual_corpus, predicted_corpus, bleu1_scores, bleu2_scores = [], [], [], []

for i, key in enumerate(eval_keys):
    predicted        = generate_caption(features[key])
    references       = [c.replace('startseq','').replace('endseq','').strip().split() for c in mapping[key]]
    predicted_tokens = predicted.split()
    actual_corpus.append(references)
    predicted_corpus.append(predicted_tokens)
    bleu1_scores.append(corpus_bleu([references],[predicted_tokens], weights=(1,0,0,0)))
    bleu2_scores.append(corpus_bleu([references],[predicted_tokens], weights=(0.5,0.5,0,0)))
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{len(eval_keys)} done...")

bleu1 = corpus_bleu(actual_corpus, predicted_corpus, weights=(1.0, 0,    0,    0   ))
bleu2 = corpus_bleu(actual_corpus, predicted_corpus, weights=(0.5, 0.5,  0,    0   ))
bleu3 = corpus_bleu(actual_corpus, predicted_corpus, weights=(0.33,0.33, 0.33, 0   ))
bleu4 = corpus_bleu(actual_corpus, predicted_corpus, weights=(0.25,0.25, 0.25, 0.25))

print(f"\nBLEU-1: {bleu1*100:.2f}% | BLEU-2: {bleu2*100:.2f}% | BLEU-3: {bleu3*100:.2f}% | BLEU-4: {bleu4*100:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(14,5))
fig.suptitle('VisionVoice — Validation Accuracy', fontsize=14)
scores = [bleu1*100, bleu2*100, bleu3*100, bleu4*100]
labels = ['BLEU-1','BLEU-2','BLEU-3','BLEU-4']
colors = ['#2ecc71','#3498db','#9b59b6','#e74c3c']
bars   = axes[0].bar(labels, scores, color=colors, width=0.5)
axes[0].set_ylim(0,100)
for bar, score in zip(bars, scores):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{score:.1f}%', ha='center', fontsize=11, fontweight='bold')
axes[1].hist(np.array(bleu1_scores)*100, bins=20, color='#3498db', edgecolor='white', alpha=0.8)
axes[1].axvline(x=np.mean(bleu1_scores)*100, color='red', linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR,'validation_results.png'), dpi=150)
plt.show()
print("✅ Chart saved!")
