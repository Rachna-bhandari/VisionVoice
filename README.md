VisionVoice 🎙️
AI-Powered Image Captioning with Live Camera & Audio Output

What is VisionVoice?
VisionVoice is an AI system that:

Sees — Takes a photo using your webcam (or loads an existing image)

Understands — Uses a trained Deep Learning model (InceptionV3 + LSTM) to generate a natural language caption

Speaks — Reads the caption out loud using text-to-speech


Project Structure
VisionVoice/
├── data/

│   ├── Images/              ← Dataset images + captured photos saved here

│   └── captions.txt         ← Flickr8k-style captions file

│

├── extract_features.py      ← Step 1: Extract CNN features from images

├── preprocess.py            ← Step 2: Tokenize and prepare captions

├── model_structure.py       ← Model architecture reference / test file

├── train.py                 ← Step 3: Train the caption model

├── camera.py                ← OpenCV live camera capture module

├── voice.py                 ← Text-to-speech helper (pyttsx3)

├── main.py                  ← Step 4: Run the full demo

How the Model Works

[Image] → InceptionV3 (CNN) → Feature Vector (2048,)
                                        ↓
                               Encoder Dense (512)
                                        ↓
[Caption so far] → Embedding → LSTM → Decoder → Next Word
                                        ↓
                               Repeat until 'endseq'

InceptionV3 extracts visual features from the image

LSTM generates the caption word by word

pyttsx3 speaks the final caption out loud

OpenCV handles live webcam capture


Requirements

Install all dependencies with:

bashpip install tensorflow numpy matplotlib pyttsx3 tqdm opencv-python

PackagePurposetensorflowModel training & inferencenumpyArray operationsmatplotlibDisplay image with captionpyttsx3Text-to-speech audio outputtqdmProgress 
bars during feature extractionopencv-pythonLive webcam capture

Step-by-Step Setup & Run

Step 1 — Extract Image Features
bashpython extract_features.py
Reads all images from data/Images/ and saves CNN features to features.pkl

Step 2 — Preprocess Captions
bashpython preprocess.py
Reads data/captions.txt, tokenizes all captions, saves tokenizer.pkl

Step 3 — Train the Model
bashpython train.py

Trains a 512-neuron LSTM model for 30 epochs
Saves the best model as vision_voice_model.keras
Automatically resumes training if a saved model already exists
Target loss: below 3.0


Step 4 — Run the Demo
bashpython main.py
You will see this menu:
=============================================
       VisionVoice PBL Project
=============================================
