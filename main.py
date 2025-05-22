from deepface import DeepFace
import cv2
import mediapipe as mp
from collections import Counter

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

# Inicializa MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Atividades básicas
atividade_summary = []

def classificar_atividade(landmarks):
    if not landmarks:
        return "desconhecida"

    # Exemplo de lógica simples: se mãos estão acima da cabeça → "braços levantados"
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

    if left_wrist.y < nose.y and right_wrist.y < nose.y:
        return "PRA CIMA"
    elif left_wrist.y > nose.y and right_wrist.y > nose.y:
        return "PRA BAIXO"
    else:
        return "movimento indefinido"

# Captura da webcam
cap = cv2.VideoCapture(0)
frame_count = 0
emotion_summary = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # EMOÇÃO
    try:
        result = DeepFace.analyze(rgb, actions=['emotion'], enforce_detection=False)
        results = result if isinstance(result, list) else [result]

        for face_data in results:
            emotion_en = face_data.get('dominant_emotion', '')
            emotion = EMOCOES_PT.get(emotion_en, emotion_en)
            emotion_summary.append(emotion)

            region = face_data.get('region', {})
            x, y, w, h = region.get('x'), region.get('y'), region.get('w'), region.get('h')

            if None not in (x, y, w, h):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f'{emotion}', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    except Exception as e:
        print("Erro na análise de emoção:", e)

    # ATIVIDADE com MediaPipe Pose
    pose_result = pose.process(rgb)
    if pose_result.pose_landmarks:
        mp_draw.draw_landmarks(frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        atividade = classificar_atividade(pose_result.pose_landmarks.landmark)
        atividade_summary.append(atividade)

        # Exibir atividade na imagem
        cv2.putText(frame, f'Atividade: {atividade}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Exibe a janela
    frame_count += 1
    cv2.imshow("Detect de emocao e Atividades", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# GERAÇÃO DE RESUMO
print("\n📊 RESUMO FINAL")

print("\n🎭 Emoções detectadas:")
for emotion, count in Counter(emotion_summary).items():
    print(f"- {emotion}: {count} vez(es)")

print("\n🏃‍♂️ Atividades detectadas:")
for atividade, count in Counter(atividade_summary).items():
    print(f"- {atividade}: {count} vez(es)")
print("\n🔚 Fim do programa.")