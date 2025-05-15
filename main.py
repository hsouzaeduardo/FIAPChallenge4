from deepface import DeepFace
import cv2

cap = cv2.VideoCapture('video.mp4')
frame_count = 0
emotion_summary = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        emotion_summary.append(emotion)
    except:
        continue

    frame_count += 1

cap.release()

# Gerar relatório simples
from collections import Counter
report = Counter(emotion_summary)

with open("report.txt", "w") as f:
    f.write(f"Total de frames analisados: {frame_count}\n")
    f.write("Emoções detectadas:\n")
    for emotion, count in report.items():
        f.write(f"{emotion}: {count}\n")
