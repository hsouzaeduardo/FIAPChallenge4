from deepface import DeepFace
import cv2


# Tradução das emoções para português
EMOCOES_PT = {
    'angry': 'raiva',
    'disgust': 'nojo',
    'fear': 'medo',
    'happy': 'feliz',
    'sad': 'triste',
    'surprise': 'surpreso',
    'neutral': 'neutro'
}

cap = cv2.VideoCapture(0)
frame_count = 0
emotion_summary = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionar o frame para acelerar processamento (opcional)
    frame_resized = cv2.resize(frame, (640, 480))

    try:
        result = DeepFace.analyze(frame_resized, actions=['emotion'], enforce_detection=False)

        # Trata os dois casos: lista ou dict
        results = result if isinstance(result, list) else [result]

        for face_data in results:
            # Emoção
            emotion_en = face_data.get('dominant_emotion', '')
            emotion = EMOCOES_PT.get(emotion_en, emotion_en)
            emotion_summary.append(emotion)

            # Bounding box (x, y, w, h)
            region = face_data.get('region', {})
            x, y, w, h = region.get('x'), region.get('y'), region.get('w'), region.get('h')

            if None not in (x, y, w, h):
                cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame_resized, f'{emotion}', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    except Exception as e:
        print("Erro:", e)
        pass

    frame_count += 1
    cv2.imshow("Detecção de Emoções", frame_resized)

    # Pressione 'ESC' para sair
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
