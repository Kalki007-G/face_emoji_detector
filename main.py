import cv2
import numpy as np
from deepface import DeepFace

# Load emojis
def load_emoji(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

emoji_map = {
    'happy': load_emoji('emoji/happy.png'),
    'sad': load_emoji('emoji/sad.png'),
    'surprise': load_emoji('emoji/surprise.png'),
    'neutral': load_emoji('emoji/neutral.png'),
    'angry': load_emoji('emoji/thinking.png'),
    'fear': load_emoji('emoji/thinking.png'),
    'disgust': load_emoji('emoji/thinking.png')
}

# Overlay emoji with alpha blending
def overlay_emoji(frame, emoji, x, y, w, h):
    emoji = cv2.resize(emoji, (w, h))
    y1, y2 = y, y + emoji.shape[0]
    x1, x2 = x, x + emoji.shape[1]

    if y2 > frame.shape[0] or x2 > frame.shape[1]:
        return frame  # Avoid overflow

    alpha_emoji = emoji[:, :, 3] / 255.0
    alpha_frame = 1.0 - alpha_emoji

    for c in range(3):
        frame[y1:y2, x1:x2, c] = (
            alpha_emoji * emoji[:, :, c] + alpha_frame * frame[y1:y2, x1:x2, c]
        )
    return frame

cap = cv2.VideoCapture(0)

print("🔍 Starting face emoji detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion']

        emoji_img = emoji_map.get(dominant_emotion)
        if emoji_img is not None:
            frame = overlay_emoji(frame, emoji_img, 100, 100, 150, 150)

        cv2.putText(frame, f"Emotion: {dominant_emotion}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except Exception as e:
        print("⚠️ Error detecting emotion:", e)

    cv2.imshow("Face Emoji Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
