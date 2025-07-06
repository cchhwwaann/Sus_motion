import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque

# 모델 로딩 (절대 경로 사용)
model = load_model('C:/Users/ghksc/Desktop/project/motion_screen/2.15/gesture_model.h5')

# 클래스 이름 (라벨 이름과 동일하게 0과 1)
gestures = ['0', '1']

# 슬라이딩 윈도우 설정
window_size = 30       # 모델이 요구하는 프레임 수
step_size = 2          # 예측 업데이트 간격을 더 빠르게 (2 프레임마다)
change_threshold = 2   # 연속해서 2회 이상 다른 예측이 나오면 동작 전환으로 판단

# Mediapipe Hands 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# 웹캠 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

# 프레임 버퍼 (최신 30프레임 유지)
frame_buffer = deque(maxlen=window_size)
frame_count = 0
last_prediction = None
change_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            if len(landmarks) == 63:  # 21개 랜드마크 * 3 좌표
                frame_buffer.append(landmarks)
    else:
        # 손이 감지되지 않으면 버퍼 초기화
        frame_buffer.clear()

    frame_count += 1

    # 충분한 프레임이 쌓였고, 지정 간격마다 예측 수행
    if len(frame_buffer) == window_size and (frame_count % step_size == 0):
        input_data = np.array(frame_buffer)             # (30, 63)
        input_data = np.expand_dims(input_data, axis=0)     # (1, 30, 63)
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]

        # 초기 예측 설정
        if last_prediction is None:
            last_prediction = predicted_class
            change_counter = 0
        else:
            # 만약 예측 결과가 이전과 다르면 카운터 증가
            if predicted_class != last_prediction:
                change_counter += 1
            else:
                change_counter = 0

        # 연속 변화 감지 시 동작 전환 처리
        if change_counter >= change_threshold:
            last_prediction = predicted_class
            change_counter = 0
            frame_buffer.clear()  # 버퍼 클리어하여 새 동작 프레임만 쌓임

        cv2.putText(frame, f"Gesture: {gestures[last_prediction]} ({confidence:.2f})",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        print(f"Predicted: {gestures[last_prediction]}, Confidence: {confidence:.2f}")

    cv2.imshow("Real-Time Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
