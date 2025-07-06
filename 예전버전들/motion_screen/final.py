import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyautogui
from screeninfo import get_monitors
import ctypes
# 모델 로드
model = tf.keras.models.load_model('hand_gesture_model.h5')
# 미디어파이프 설정
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
# smoothing을 위한 이전 마우스 좌표 (초기값은 화면 중앙)
prev_cursor = (screen_width // 2, screen_height // 2)
smoothing_factor = 0.8  # 0~1 사이, 클수록 급격한 변화 반영
# 확대(감도 조절) 기능을 위한 파라미터
ref_hand_size = 1000  # 기준 손 크기 (픽셀). 이보다 작으면 확대(감도 증가)
max_amplification = 4.0  # 최대 확대(감도) 배율
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    # 매 3프레임마다 처리 (과부하 방지)
    if frame_count % 3 != 0:
        continue
    # 좌우 반전 및 색상 변환
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    detected_gestures = []
    pointer_positions = []  # 각 손마다 계산된 마우스 좌표
    frame_height, frame_width, _ = frame.shape
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # 모델 예측
            landmarks = hand_landmarks.landmark
            input_data = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten().reshape(1, -1)
            prediction = model.predict(input_data)
            gesture = int(np.argmax(prediction))
            detected_gestures.append(gesture)
            # 손의 크기를 계산 (정규화 좌표를 이용하여 바운딩 박스 크기 산출)
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            hand_width = (max_x - min_x) * frame_width
            hand_height = (max_y - min_y) * frame_height
            hand_size = max(hand_width, hand_height)
            # 인덱스 손가락 끝(8번 랜드마크)의 정규화 좌표
            index_x = hand_landmarks.landmark[8].x
            index_y = hand_landmarks.landmark[8].y
            # 기본적으로는 인덱스 손가락의 정규화 좌표를 그대로 사용.
            # 다만, 손이 작으면 (즉, 사용자가 멀리 있으면) 화면 중앙(0.5,0.5)에서의
            # 오프셋을 확대하여 커서 이동 범위를 넓힙니다.
            if hand_size > 0 and hand_size < ref_hand_size:
                amplification = ref_hand_size / hand_size
                # 최대 확대 배율 제한
                amplification = min(amplification, max_amplification)
            else:
                amplification = 1.0
            # 화면 중앙을 기준으로 오프셋 확대 적용
            # (0.5, 0.5)를 기준으로 좌표 오프셋를 확대하여 전체 0~1 범위로 매핑
            cursor_norm_x = 0.5 + (index_x - 0.5) * amplification
            cursor_norm_y = 0.5 + (index_y - 0.5) * amplification
            # 0~1 범위로 클램핑
            cursor_norm_x = max(0, min(cursor_norm_x, 1))
            cursor_norm_y = max(0, min(cursor_norm_y, 1))
            # 화면 좌표 변환
            pointer_x = int(cursor_norm_x * screen_width)
            pointer_y = int(cursor_norm_y * screen_height)
            # smoothing 적용
            smooth_x = int(smoothing_factor * pointer_x + (1 - smoothing_factor) * prev_cursor[0])
            smooth_y = int(smoothing_factor * pointer_y + (1 - smoothing_factor) * prev_cursor[1])
            prev_cursor = (smooth_x, smooth_y)
            pointer_positions.append(prev_cursor)
        # 두 손 제스처 조합에 따른 클릭 처리
        if 0 in detected_gestures and 1 in detected_gestures:
            pyautogui.click()
        elif 0 in detected_gestures and 2 in detected_gestures:
            pyautogui.rightClick()
        else:
            # 여러 손이 있다면 첫 번째 손의 좌표를 사용하여 커서 이동
            if pointer_positions:
                ctypes.windll.user32.SetCursorPos(pointer_positions[0][0], pointer_positions[0][1])
            # 개별 제스처에 따른 추가 동작 처리
            for gesture, pos in zip(detected_gestures, pointer_positions):
                if gesture == 3:
                    pyautogui.press('volumeup')
                elif gesture == 4:
                    pyautogui.press('volumedown')
                elif gesture == 5:
                    pyautogui.press('playpause')
                elif gesture == 6:
                    pyautogui.hotkey('alt', 'left')
    # 결과 프레임 출력 (화면 확대 효과 없이 원본 프레임 표시)
    cv2.imshow('Hand Gesture Control', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()