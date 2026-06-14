"""
VisionVoice — voice.py
Offline text-to-speech using pyttsx3.
Speaks the generated caption aloud.
"""

import pyttsx3


def speak_caption(caption: str):
    """
    Cleans the caption (removes startseq / endseq tokens) and speaks it aloud.
    """
    clean = (caption
             .replace('startseq', '')
             .replace('endseq', '')
             .strip())

    if not clean:
        print("[voice] Caption is empty — nothing to speak.")
        return

    print(f"\n🔊 VisionVoice says: \"{clean}\"\n")

    engine = pyttsx3.init()
    engine.setProperty('rate',   160)   # words per minute
    engine.setProperty('volume', 1.0)   # 0.0 – 1.0

    # Optional: pick a clearer voice if multiple are available
    voices = engine.getProperty('voices')
    if voices:
        engine.setProperty('voice', voices[0].id)

    engine.say(clean)
    engine.runAndWait()


if __name__ == "__main__":
    speak_caption("a dog is running on the grass")
