import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyautogui
from collections import deque
from screeninfo import get_monitors
import ctypes
import time

# --- 모델 로드 ---
static_model = tf.keras.models.load_model('hand_gesture_model_static.h5')
dynamic_model = tf.keras.models.load_model('hand_gesture_model_dynamic.h5')

# --- Mediapipe 초기화 ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- 화면 및 웹캠 설정 ---
screen = get_monitors()[0]
screen_width, screen_height = screen.width, screen.height
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# --- 스무딩 및 센서티비티 설정 ---
prev_cursor = (screen_width // 2, screen_height // 2)
smoothing_factor = 0.8
high_sensitivity = 2.0
low_sensitivity = 1.0
min_hand_size = 50
max_hand_size = 150

# --- 동적 제스처를 위한 슬라이딩 윈도우 ---
DYNAMIC_WINDOW_SIZE = 42  # 모델이 51 프레임을 기대한다고 가정
dynamic_window = deque(maxlen=DYNAMIC_WINDOW_SIZE)
last_click_time = 0
CLICK_COOLDOWN = 1.0  # 클릭 후 최소 1초 대기

# 동적 클릭이 이미 발생했는지 여부를 추적하는 플래그
dynamic_click_triggered = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    frame_height, frame_width, _ = frame.shape

    right_hand_found = False
    right_hand_landmarks = None

    # 오른손만 처리
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            if handedness.classification[0].label == "Right":
                right_hand_found = True
                right_hand_landmarks = hand_landmarks
                break

    if right_hand_found and right_hand_landmarks is not None:
        mp_drawing.draw_landmarks(frame, right_hand_landmarks, mp_hands.HAND_CONNECTIONS)
        landmarks = [[lm.x, lm.y, lm.z] for lm in right_hand_landmarks.landmark]
        flattened_landmarks = np.array(landmarks).flatten().tolist()  # 길이 63

        # --- 정적 제스처 예측 (커서 이동) ---
        static_input = np.array(flattened_landmarks).reshape(1, -1)
        static_pred = static_model.predict(static_input)
        static_gesture = int(np.argmax(static_pred))
        if static_gesture == 0:
            xs = [pt[0] for pt in landmarks]
            ys = [pt[1] for pt in landmarks]
            hand_width = (max(xs) - min(xs)) * frame_width
            hand_height = (max(ys) - min(ys)) * frame_height
            hand_size = max(hand_width, hand_height)
            if hand_size <= min_hand_size:
                dynamic_sensitivity = high_sensitivity
            elif hand_size >= max_hand_size:
                dynamic_sensitivity = low_sensitivity
            else:
                ratio = (hand_size - min_hand_size) / (max_hand_size - min_hand_size)
                dynamic_sensitivity = high_sensitivity - ratio * (high_sensitivity - low_sensitivity)
            index_x = right_hand_landmarks.landmark[8].x
            index_y = right_hand_landmarks.landmark[8].y
            new_norm_x = 0.5 + (index_x - 0.5) * dynamic_sensitivity
            new_norm_y = 0.5 + (index_y - 0.5) * dynamic_sensitivity
            new_norm_x = max(0, min(new_norm_x, 1))
            new_norm_y = max(0, min(new_norm_y, 1))
            pointer_x = int(new_norm_x * screen_width)
            pointer_y = int(new_norm_y * screen_height)
            smooth_x = int(smoothing_factor * pointer_x + (1 - smoothing_factor) * prev_cursor[0])
            smooth_y = int(smoothing_factor * pointer_y + (1 - smoothing_factor) * prev_cursor[1])
            prev_cursor = (smooth_x, smooth_y)
            ctypes.windll.user32.SetCursorPos(smooth_x, smooth_y)
            cv2.putText(frame, "Static: Move Pointer", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"Static: {static_gesture}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # --- 동적 제스처 슬라이딩 윈도우 업데이트 ---
        dynamic_window.append(flattened_landmarks)
        if len(dynamic_window) == DYNAMIC_WINDOW_SIZE:
            try:
                dynamic_input = np.array(dynamic_window).reshape(1, DYNAMIC_WINDOW_SIZE, 63)
                dynamic_pred = dynamic_model.predict(dynamic_input)
                dynamic_gesture = int(np.argmax(dynamic_pred))
                cv2.putText(frame, f"Dynamic: {dynamic_gesture}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                # 만약 동적 제스처가 0이면 클릭을 시도
                if dynamic_gesture == 1:
                    if not dynamic_click_triggered:
                        current_time = time.time()
                        if current_time - last_click_time > CLICK_COOLDOWN:
                            pyautogui.click()
                            last_click_time = current_time
                            dynamic_click_triggered = True
                            cv2.putText(frame, "Click!", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    # 동적 제스처가 0이 아니면 클릭 플래그 초기화
                    dynamic_click_triggered = False

            except Exception as e:
                print("Dynamic gesture prediction error:", e)
    else:
        cv2.putText(frame, "No Right Hand", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Hand Gesture Control', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
