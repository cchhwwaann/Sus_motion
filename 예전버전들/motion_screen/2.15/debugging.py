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

# 웹캠 실행 및 낮은 해상도 설정
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

frame_count = 0

# smoothing을 위한 이전 마우스 좌표 (초기값은 화면 중앙)
prev_cursor = (screen_width // 2, screen_height // 2)
smoothing_factor = 0.8  # 0~1 사이, 클수록 갑작스러운 변화 반영

# 확대(zoom) 기능을 위한 파라미터
ref_hand_size = 100  # 이상적인 손의 크기(픽셀) – 이보다 작으면 확대
max_scale = 3.0      # 최대 확대 배율

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
    pointer_positions = []  # 각 손마다 계산된 마우스 좌표(zoom 적용)

    # frame의 실제 크기 (320x240)
    frame_height, frame_width, _ = frame.shape

    if results.multi_hand_landmarks:
        # 여러 손이 감지되면, 각 손에 대해 처리합니다.
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 손 랜드마크 추출 (정규화된 좌표: 0~1)
            landmarks = hand_landmarks.landmark
            # 모델 입력 데이터 생성 (원본과 동일)
            input_data = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten().reshape(1, -1)
            prediction = model.predict(input_data)
            gesture = int(np.argmax(prediction))
            detected_gestures.append(gesture)

            # --- 여기서부터 확대(zoom) 적용 ---
            # 모든 x, y 좌표 리스트
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            # 손의 바운딩 박스 (정규화 좌표)
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            # 픽셀 단위 손 크기
            hand_width = (max_x - min_x) * frame_width
            hand_height = (max_y - min_y) * frame_height
            hand_size = max(hand_width, hand_height)

            # 손이 작으면(멀리 있으면) 확대 배율 계산 (최대 max_scale까지)
            if hand_size < ref_hand_size and hand_size > 0:
                scale = ref_hand_size / hand_size
                scale = min(scale, max_scale)
            else:
                scale = 1.0

            # 손 중심 (정규화 좌표)
            center_x = np.mean(xs)
            center_y = np.mean(ys)

            # 검지 손가락 끝(8번 랜드마크)의 정규화 좌표
            index_x = hand_landmarks.landmark[8].x
            index_y = hand_landmarks.landmark[8].y

            # 확대 적용: 손 중심을 기준으로 검지 좌표와의 차이를 확대
            new_norm_x = center_x + (index_x - center_x) * scale
            new_norm_y = center_y + (index_y - center_y) * scale
            # 0~1 범위 클램핑
            new_norm_x = max(0, min(new_norm_x, 1))
            new_norm_y = max(0, min(new_norm_y, 1))
            # 화면 좌표로 변환
            pointer_x = int(new_norm_x * screen_width)
            pointer_y = int(new_norm_y * screen_height)
            # --- 확대 적용 끝 ---

            # smoothing 적용 (이전 좌표와 현재 좌표를 혼합)
            smooth_x = int(smoothing_factor * pointer_x + (1 - smoothing_factor) * prev_cursor[0])
            smooth_y = int(smoothing_factor * pointer_y + (1 - smoothing_factor) * prev_cursor[1])
            prev_cursor = (smooth_x, smooth_y)

            pointer_positions.append(prev_cursor)

        # 두 손 이상의 제스처 조합에 따른 클릭 처리
        if 0 in detected_gestures and 1 in detected_gestures:
            pyautogui.click()
        elif 0 in detected_gestures and 2 in detected_gestures:
            pyautogui.rightClick()
        else:
            # 포인터 이동: 여러 손이 있다면 첫 번째 손의 좌표 사용
            if pointer_positions:
                ctypes.windll.user32.SetCursorPos(pointer_positions[0][0], pointer_positions[0][1])

            # 개별 제스처에 따른 동작 처리 (예, 볼륨 조절 등)
            for gesture, pos in zip(detected_gestures, pointer_positions):
                # (포인터 이동은 이미 위에서 처리하므로 여기서는 추가 동작만)
                if gesture == 3:
                    pyautogui.press('volumeup')
                elif gesture == 4:
                    pyautogui.press('volumedown')
                elif gesture == 5:
                    pyautogui.press('playpause')
                elif gesture == 6:
                    pyautogui.hotkey('alt', 'left')

    # 화면에 결과 출력
    cv2.imshow('Hand Gesture Control', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()