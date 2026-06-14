"""
VisionVoice — voice.py
Offline text-to-speech using pyttsx3.
"""

import pyttsx3

def speak_caption(caption: str):
    clean = caption.replace('startseq', '').replace('endseq', '').strip()
    if not clean:
        print("[voice] Caption is empty.")
        return
    print(f"\n🔊 VisionVoice says: \"{clean}\"\n")
    engine = pyttsx3.init()
    engine.setProperty('rate',   160)
    engine.setProperty('volume', 1.0)
    voices = engine.getProperty('voices')
    if voices:
        engine.setProperty('voice', voices[0].id)
    engine.say(clean)
    engine.runAndWait()

if __name__ == "__main__":
    speak_caption("a dog is running on the grass")
