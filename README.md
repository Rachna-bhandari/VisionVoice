<div align="center">

# 🎙️ VisionVoice
### AI-Powered Image Captioning with Live Camera & Audio Output

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-InceptionV3%20%2B%20LSTM-red?style=for-the-badge&logo=keras&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

<br/>

> **VisionVoice** is a deep learning system that clicks a live photo via webcam,  
> generates a natural language caption using **InceptionV3 + LSTM**, and **speaks it aloud** — all in real time.

<br/>

</div>

---

## 📌 Table of Contents

- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Pipeline Overview](#-pipeline-overview)
- [Camera Controls](#-camera-controls)
- [File Generation Summary](#-file-generation-summary)

---

## 🧠 How It Works

```
┌──────────────┐    ┌───────────────────┐    ┌─────────-─────────────┐
│   Webcam     │ ▶     InceptionV3 CNN  │ ▶    Feature Vector       │
│  (OpenCV)    │    │  (ImageNet)       │    │  (2048 dimensions)    │
└──────────────┘    └───────────────────┘    └─────────-─┬───────────┘
                                                         │
                                                         ▼
                                             ┌──────────────────────┐
                                             │  Encoder Dense (512) │
                                             └──────────┬───────────┘
                                                         │
┌──────────────┐    ┌───────────────────┐                │
│  [startseq]  │───▶│  Embedding (512)  │───────────────┘
│  (seed word) │    │  + LSTM (512)     │       (merged via add)
└──────────────┘    └───────────────────┘               │
                                                        ▼
                                             ┌──────────────────────┐
                                             │  Decoder Dense (512) │
                                             │  → softmax → word    │
                                             └──────────┬───────────┘
                                                        │
                                              Repeat until 'endseq'
                                                        │
                                                        ▼
                                             ┌──────────────────────┐
                                             │  pyttsx3  →  Speak   │
                                             └──────────────────────┘
```

---

## 📁 Project Structure

```
VisionVoice/
│
├── 📂 data/
│   ├── 📂 Images/              ← Dataset images + captured.jpg saved here
│   └── 📄 captions.txt         ← Flickr8k-style captions file
│
├── 📄 extract_features.py      ← Step 1 : Extract InceptionV3 features
├── 📄 preprocess.py            ← Step 2 : Tokenize captions, save mapping
├── 📄 model_structure.py       ← Reference: print model summary anytime
├── 📄 train.py                 ← Step 3 : Train the LSTM caption model
├── 📄 camera.py                ← OpenCV webcam module (capture on SPACEBAR)
├── 📄 voice.py                 ← pyttsx3 text-to-speech helper
├── 📄 main.py                  ← Step 4 : Run the live demo end-to-end
```

---

## 🛠️ Tech Stack

| Technology | Role |
|---|---|
| **Python 3.8+** | Core language |
| **TensorFlow / Keras** | Model training & inference |
| **InceptionV3** | CNN image feature extractor (pretrained, ImageNet) |
| **LSTM** | Sequence model for caption generation |
| **OpenCV** | Live webcam capture with countdown |
| **pyttsx3** | Offline text-to-speech engine |
| **Matplotlib** | Display captured image with generated caption |
| **tqdm** | Progress bar during feature extraction |

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/VisionVoice.git
cd VisionVoice
```

**2. Install all dependencies**
```bash
pip install tensorflow numpy matplotlib pyttsx3 tqdm opencv-python
```

**3. Prepare the dataset**
- Download **Flickr8k** from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- Place all images inside → `data/Images/`
- Place `captions.txt` inside → `data/`

---

## 🚀 How to Run

Run the steps **in order, once**. After that, only `main.py` is needed every time.

### Step 1 — Extract Features
```bash
python extract_features.py
```
Reads every image in `data/Images/` through InceptionV3 and saves 2048-dim vectors.  
**Output:** `features.pkl`

---

### Step 2 — Preprocess Captions
```bash
python preprocess.py
```
Cleans captions, fits a tokenizer, computes max caption length.  
**Output:** `tokenizer.pkl`, `mapping.pkl`, `max_length.txt`

---

### Step 3 — Train the Model
```bash
python train.py
```
Trains for up to 30 epochs with early stopping. Resumes automatically if interrupted.  
**Output:** `vision_voice_model.keras`  
**Target loss:** below 3.0

---

### Step 4 — Run the Live Demo
```bash
python main.py
```

| Step | What Happens |
|:----:|---|
| 1 | Webcam opens with live 720p preview |
| 2 | 3-second countdown starts on SPACEBAR |
| 3 | Photo captured & saved to `data/Images/captured.jpg` |
| 4 | InceptionV3 extracts features from the photo |
| 5 | LSTM decodes caption word-by-word |
| 6 | Image displayed with caption as title |
| 7 | Caption spoken aloud via pyttsx3 |

---

## 📷 Camera Controls

| Key | Action |
|:---:|---|
| `SPACEBAR` | Start 3-second countdown → capture photo |
| `Q` | Quit camera without capturing |

---

## 📊 Pipeline Overview

```
extract_features.py  →  features.pkl
preprocess.py        →  tokenizer.pkl + mapping.pkl + max_length.txt
train.py             →  vision_voice_model.keras
main.py              →  loads all of the above + camera.py + voice.py
```

---

## 🗂️ File Generation Summary

| File | Created By | Used By |
|---|---|---|
| `features.pkl` | `extract_features.py` | `train.py`, `main.py` |
| `tokenizer.pkl` | `preprocess.py` | `train.py`, `main.py` |
| `mapping.pkl` | `preprocess.py` | `train.py` |
| `max_length.txt` | `preprocess.py` | `train.py`, `main.py` |
| `vision_voice_model.keras` | `train.py` | `main.py` |
| `data/Images/captured.jpg` | `camera.py` | `main.py` |

---

---

<div align="center">

⭐ **If you found this project helpful, give it a star!** ⭐

</div>
