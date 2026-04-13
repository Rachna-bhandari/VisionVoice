# 🖼️ Image Caption Generator using Gemma 4

A modern **multimodal image captioning project** built using **Google DeepMind's Gemma 4 Vision model**.
This project generates **human-like natural language captions directly from images** using a **single multimodal transformer model**.

---

## 🚀 Project Overview

This project uses **Gemma 4** for direct **image-to-text caption generation**.
The model takes an image as input and produces a detailed, context-aware caption in natural language.

```text
Image → Gemma 4 → Caption
```

It is designed for:

* automatic image description
* accessibility support
* dataset annotation
* multimodal AI research
* real-world caption generation applications

---

## ✨ Features

* 📷 Direct image-to-text caption generation
* 🧠 Powered by **Gemma 4 E4B-it multimodal model**
* 📝 Detailed and natural captions
* 🌍 Multilingual caption generation support
* 🔍 Strong scene understanding
* 💻 Local inference support

---

## 🛠️ Tech Stack

* **Python 3.10+**
* **Transformers**
* **PyTorch**
* **Pillow**
* **Gemma 4 E4B-it**

---

## 📂 Project Structure

```text
Gemma4-Image-Captioning/
│
├── app.py                 # Main inference script
├── requirements.txt       # Dependencies
├── sample_images/         # Test images
├── outputs/               # Generated captions
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/gemma4-image-captioning.git
cd gemma4-image-captioning
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch transformers pillow accelerate
```

---

## ▶️ Usage

```python
from transformers import AutoProcessor, AutoModelForMultimodalLM
from PIL import Image

MODEL_ID = "google/gemma-4-E4B-it"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto"
)

image = Image.open("sample_images/dog.jpg")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Write a detailed caption for this image."}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)

input_len = inputs["input_ids"].shape[-1]
outputs = model.generate(**inputs, max_new_tokens=100)
caption = processor.decode(outputs[0][input_len:], skip_special_tokens=True)

print(caption)
```

---

## 📌 Example Output

### Input Image

A dog running in grass.

### Generated Caption

> A brown dog is running joyfully across a green grassy field during daytime.

---

## 📈 Future Improvements

* Fine-tune on custom datasets
* Add voice caption output
* Deploy with Streamlit / Gradio
* Build real-time webcam captioning
* Add multilingual caption generation

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a pull request

---

## 📜 License

This project is licensed under the **Apache 2.0 License**.

---

## 🙌 Acknowledgements

* Google DeepMind for **Gemma 4**
* Hugging Face Transformers
* Open-source multimodal AI community

---

## ⭐ Support

If you found this project useful, please **star the repository ⭐**.
# app-voice
