import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyautogui
from screeninfo import get_monitors
import ctypes

# 학습된 모델 로드 (TensorFlow/Keras)
model = tf.keras.models.load_model('hand_gesture_model.h5')

# 미디어파이프 손 인식 설정
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 화면 크기 가져오기
screen = get_monitors()[0]
screen_width, screen_height = screen.width, screen.height

# 웹캠 실행 및 낮은 해상도 설정 (속도 개선 목적)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

frame_count = 0

# 스무딩을 위한 이전 마우스 좌표 (초기값: 화면 중앙)
prev_cursor = (screen_width // 2, screen_height // 2)
smoothing_factor = 0.8  # 0~1 사이 값, 작을수록 부드럽게 이동

# 센서티비티 관련 기본값 (동적 조절)
high_sensitivity = 2.0   # 손이 멀리 있을 때(크기가 작을 때) 적용할 sensitivity
low_sensitivity = 1.0    # 손이 가까울 때(크기가 클 때) 적용할 sensitivity
min_hand_size = 50       # 손 크기가 이보다 작으면 far 상태로 간주
max_hand_size = 150      # 손 크기가 이보다 크면 close 상태로 간주

# **상태 변수 (왼쪽 클릭 인식을 위한 상태 전이)**
# 초기 상태는 0 (마우스 이동 상태)로 가정합니다.
prev_main_gesture = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # 매 3프레임마다 처리하여 연산 부담 줄이기
    if frame_count % 3 != 0:
        continue

    # 좌우 반전 및 BGR→RGB 변환
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    detected_gestures = []   # 각 손마다 예측한 제스처 (정수 값)
    pointer_positions = []   # 각 손마다 계산된 마우스 좌표 (스무딩 적용)

    frame_height, frame_width, _ = frame.shape  # (320x240)

    # 오른손만 인식하도록 multi_handedness도 함께 사용
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # handedness 정보에서 label 추출 ("Left" 또는 "Right")
            hand_label = handedness.classification[0].label
            if hand_label != "Right":
                continue

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 손의 21개 랜드마크 (정규화된 좌표: 0~1)
            landmarks = hand_landmarks.landmark
            # 모델 입력 데이터 구성 (각 랜드마크의 x, y, z를 flatten)
            input_data = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten().reshape(1, -1)
            prediction = model.predict(input_data)
            gesture = int(np.argmax(prediction))
            detected_gestures.append(gesture)

            # --- 동적 sensitivity 적용을 위한 손 크기 계산 ---
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            hand_width = (max_x - min_x) * frame_width
            hand_height = (max_y - min_y) * frame_height
            hand_size = max(hand_width, hand_height)

            # 손 크기에 따라 sensitivity 조절 (선형 보간)
            if hand_size <= min_hand_size:
                dynamic_sensitivity = high_sensitivity
            elif hand_size >= max_hand_size:
                dynamic_sensitivity = low_sensitivity
            else:
                ratio = (hand_size - min_hand_size) / (max_hand_size - min_hand_size)
                dynamic_sensitivity = high_sensitivity - ratio * (high_sensitivity - low_sensitivity)
            # --- 동적 sensitivity 적용 끝 ---

            # 검지 손가락 끝(8번 랜드마크)의 정규화 좌표 사용
            index_x = hand_landmarks.landmark[8].x
            index_y = hand_landmarks.landmark[8].y

            # 센터(0.5, 0.5)를 기준으로, 동적으로 계산된 sensitivity를 적용하여 좌표 보정
            new_norm_x = 0.5 + (index_x - 0.5) * dynamic_sensitivity
            new_norm_y = 0.5 + (index_y - 0.5) * dynamic_sensitivity

            # 0~1 범위 내로 클램핑
            new_norm_x = max(0, min(new_norm_x, 1))
            new_norm_y = max(0, min(new_norm_y, 1))

            # 화면 좌표로 변환
            pointer_x = int(new_norm_x * screen_width)
            pointer_y = int(new_norm_y * screen_height)

            # 스무딩 적용 (이전 좌표와 현재 좌표 혼합)
            smooth_x = int(smoothing_factor * pointer_x + (1 - smoothing_factor) * prev_cursor[0])
            smooth_y = int(smoothing_factor * pointer_y + (1 - smoothing_factor) * prev_cursor[1])
            prev_cursor = (smooth_x, smooth_y)

            pointer_positions.append(prev_cursor)

        # 오른손이 하나라도 감지된 경우에만 처리
        if detected_gestures:
            # **메인 손**: 첫 번째 손의 제스처를 기준으로 처리
            main_gesture = detected_gestures[0]

            # 마우스 커서 이동: **라벨 0**인 경우에만 이동합니다.
            if main_gesture == 0:
                ctypes.windll.user32.SetCursorPos(pointer_positions[0][0], pointer_positions[0][1])

            # **왼쪽 클릭 처리 (상태 전이 방식)**
            # 이전 프레임의 메인 제스처가 1(클릭 제스처)이고, 이번 프레임에 0(마우스 이동)으로 돌아오면 클릭 실행
            if prev_main_gesture == 1 and main_gesture == 0:
                pyautogui.click()

            # 추가 제스처에 따른 동작 처리
            # (주석 처리한 추가 동작들은 필요에 따라 활성화)
            # if main_gesture == 3:
            #     pyautogui.press('volumeup')
            # elif main_gesture == 4:
            #     pyautogui.press('volumedown')
            # elif main_gesture == 5:
            #     pyautogui.press('playpause')
            # elif main_gesture == 6:
            #     pyautogui.hotkey('alt', 'left')

            # 다음 프레임을 위해 메인 제스처 상태 업데이트
            prev_main_gesture = main_gesture

            # 디버깅을 위한 현재 메인 제스처와 sensitivity 값을 화면에 표시
            cv2.putText(frame, f'Gesture: {main_gesture}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Sens: {dynamic_sensitivity:.2f}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Hand Gesture Control', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
