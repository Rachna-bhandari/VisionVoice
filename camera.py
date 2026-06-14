"""
VisionVoice — camera.py
Opens webcam, captures photo on SPACEBAR with 3-sec countdown.
SPACEBAR → capture | Q → quit
"""

import cv2, time, os

CAPTURE_PATH = os.path.join("data", "Images", "captured.jpg")

def capture_image(save_path=CAPTURE_PATH):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("\n📷 Webcam ready! SPACEBAR = capture | Q = quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        display = frame.copy()
        cv2.putText(display, "SPACEBAR: Capture | Q: Quit",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("VisionVoice — Live Camera", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            cap.release()
            cv2.destroyAllWindows()
            return None

        if key == ord(' '):
            for count in range(3, 0, -1):
                deadline = time.time() + 1.0
                while time.time() < deadline:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    overlay = frame.copy()
                    cv2.putText(overlay, str(count),
                                (int(overlay.shape[1]/2)-50, int(overlay.shape[0]/2)+50),
                                cv2.FONT_HERSHEY_SIMPLEX, 8, (0,0,255), 12, cv2.LINE_AA)
                    cv2.imshow("VisionVoice — Live Camera", overlay)
                    cv2.waitKey(1)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(save_path, frame)
                print(f"✅ Photo saved → {save_path}")
                cv2.imshow("VisionVoice — Captured!", frame)
                cv2.waitKey(1500)
            break

    cap.release()
    cv2.destroyAllWindows()
    return save_path if os.path.exists(save_path) else None

if __name__ == "__main__":
    capture_image()
