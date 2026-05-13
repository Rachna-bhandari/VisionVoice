<div align="center">

# 🎙️ VisionVoice
### AI-Powered Image Captioning with Live Camera & Audio Output

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

<br/>

> **VisionVoice** is a deep learning project that clicks a live photo using your webcam, generates a natural language caption using an InceptionV3 + LSTM model, and then speaks it out loud — all in real time.

<br/>

</div>

---

## 📌 Table of Contents

- [Demo](#-demo)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Camera Controls](#-camera-controls)
- [File Generation Summary](#-file-generation-summary)
- [Troubleshooting](#-troubleshooting)
- [Author](#-author)

---

## 🎬 Demo

```
=============================================
        VisionVoice PBL Project
=============================================

📷 Camera is ON!
   ➤ Press SPACEBAR to capture the image
   ➤ Press  Q  to quit without capturing

✅ Image Captured!
💾 Image saved to: data/Images/captured.jpg

🤖 AI is analyzing the scene...

📝 Generated Caption: a dog is running on the grass near a fence

🖼️  Image window opened. Close it to hear the audio...
🔊 VisionVoice is saying: a dog is running on the grass near a fence
```

---

## 🧠 How It Works

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│   Webcam    │────▶│  InceptionV3 CNN  │────▶│  Feature Vector     │
│  (OpenCV)   │     │  (ImageNet)       │     │  (2048 dimensions)  │
└─────────────┘     └──────────────────┘     └──────────┬──────────┘
                                                         │
                                                         ▼
                                              ┌─────────────────────┐
                                              │   Encoder Dense     │
                                              │   (512 neurons)     │
                                              └──────────┬──────────┘
                                                         │
┌─────────────┐     ┌──────────────────┐                │
│  [startseq] │────▶│  Embedding +     │────────────────┘
│  (seed word)│     │  LSTM (512)      │         (merged)
└─────────────┘     └──────────────────┘                │
                                                         ▼
                                              ┌─────────────────────┐
                                              │   Decoder Dense     │
                                              │   → Next Word       │
                                              └──────────┬──────────┘
                                                         │
                                              (repeat until 'endseq')
                                                         │
                                                         ▼
                                              ┌─────────────────────┐
                                              │   pyttsx3 TTS       │
                                              │   Speak Caption     │
                                              └─────────────────────┘
```

---

## 📁 Project Structure

```
VisionVoice/
│
├── 📂 data/
│   ├── 📂 Images/              ← Dataset images + captured photos saved here
│   └── 📄 captions.txt         ← Flickr8k-style captions file
│
├── 📄 extract_features.py      ← Step 1: Extract CNN features from all images
├── 📄 preprocess.py            ← Step 2: Tokenize and prepare captions
├── 📄 model_structure.py       ← Model architecture definition & test
├── 📄 train.py                 ← Step 3: Train the LSTM caption model
├── 📄 camera.py                ← OpenCV live webcam capture module
├── 📄 voice.py                 ← pyttsx3 text-to-speech helper
├── 📄 main.py                  ← Step 4: Run the complete demo
└── 📄 README.md
```

---

## 🛠️ Tech Stack

| Technology | Role |
|---|---|
| **Python 3.8+** | Core language |
| **TensorFlow / Keras** | Model training & inference |
| **InceptionV3** | CNN image feature extractor (pretrained on ImageNet) |
| **LSTM** | Sequence model for caption generation |
| **OpenCV** | Live webcam capture |
| **pyttsx3** | Offline text-to-speech engine |
| **Matplotlib** | Display captured image with predicted caption |
| **tqdm** | Progress bars during feature extraction |

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/VisionVoice.git
cd VisionVoice
```

**2. Install dependencies**
```bash
pip install tensorflow numpy matplotlib pyttsx3 tqdm opencv-python
```

**3. Download the Flickr8k Dataset**
- Download from [Kaggle — Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- Place images inside `data/Images/`
- Place `captions.txt` inside `data/`

---

## 🚀 How to Run

Follow these steps **in order**:

### Step 1 — Extract Image Features
```bash
python extract_features.py
```
> Reads all images from `data/Images/` using InceptionV3 and saves feature vectors to `features.pkl`

---

### Step 2 — Preprocess Captions
```bash
python preprocess.py
```
> Cleans and tokenizes captions from `captions.txt`, saves vocabulary to `tokenizer.pkl`

---

### Step 3 — Train the Model
```bash
python train.py
```
> Trains the 512-neuron LSTM model for 30 epochs. Saves the best model as `vision_voice_model.keras`. Automatically resumes from checkpoint if training was interrupted.
>
> 🎯 **Target loss: below 3.0**

---

### Step 4 — Run the Live Demo
```bash
python main.py
```

**What happens step by step:**

| Step | Action |
|------|--------|
| 1 | Webcam opens with live preview |
| 2 | Press `SPACEBAR` to click the photo |
| 3 | Photo saved to `data/Images/captured.jpg` |
| 4 | InceptionV3 extracts features from the photo |
| 5 | LSTM generates a caption word by word |
| 6 | Captured image displayed with caption as title |
| 7 | Caption spoken out loud via pyttsx3 |

---

## 📷 Camera Controls

| Key | Action |
|-----|--------|
| `SPACEBAR` | Capture / Click the photo |
| `Q` | Quit camera without capturing |

---

## 📊 File Generation Summary

| File | Created By | Used By |
|------|-----------|---------|
| `features.pkl` | `extract_features.py` | `train.py`, `main.py` |
| `tokenizer.pkl` | `preprocess.py` | `train.py`, `main.py` |
| `vision_voice_model.keras` | `train.py` | `main.py` |
| `data/Images/captured.jpg` | `camera.py` | `main.py` |

---

## 🔧 Troubleshooting

<details>
<summary><b>📷 Camera not opening?</b></summary>

- Make sure your webcam is connected and not being used by another application
- Try changing the camera index in `camera.py`:
  ```python
  cap = cv2.VideoCapture(1)  # Try 1 instead of 0
  ```
</details>

<details>
<summary><b>❌ Model file not found?</b></summary>

- You must complete **Step 3** (`train.py`) before running `main.py`
- The file `vision_voice_model.keras` must exist in the root folder
</details>

<details>
<summary><b>🔇 pyttsx3 no audio output?</b></summary>

- **Linux:** Install espeak: `sudo apt install espeak`
- **Windows:** Default SAPI5 voice works automatically
- **macOS:** Default NSSpeechSynthesizer works automatically
</details>

<details>
<summary><b>📉 Poor caption quality?</b></summary>

- Ensure your training loss went **below 3.0**
- Try training for more epochs in `train.py` (increase `epochs=30`)
- Use the full Flickr8k dataset (8,000 images) for best results
</details>

---

## 👨‍💻 Author

**Deepesh**

> Built as a PBL (Project Based Learning) submission.
> Combines Computer Vision, Natural Language Processing, and Text-to-Speech into one end-to-end AI pipeline.

---

<div align="center">

⭐ **If you found this project helpful, consider giving it a star!** ⭐

</div>
