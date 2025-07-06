import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
gesture_label = None  # 현재 레이블
data = []

print("📢 원하는 손동작을 취한 후, 키보드에서 숫자를 눌러 레이블을 지정하세요.")
print("🔢 예: 0(클릭), 1(볼륨 업), 2(볼륨 다운) ...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])

            if gesture_label is not None:
                row = [gesture_label] + np.array(landmarks).flatten().tolist()
                data.append(row)

    cv2.imshow('Collect Gesture Data', frame)
    key = cv2.waitKey(1) & 0xFF

    # 숫자를 입력하면 해당 값이 레이블로 설정됨
    if ord('0') <= key <= ord('9'):
        gesture_label = int(chr(key))
        print(f"✅ 현재 레이블: {gesture_label}")

    # q를 누르면 종료
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 데이터 저장
df = pd.DataFrame(data)
df.to_csv('hand_gestures.csv', index=False, header=False)
print("📁 손동작 데이터가 'hand_gestures.csv'로 저장되었습니다.")
