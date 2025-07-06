import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyautogui
from screeninfo import get_monitors

#윈도우만
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

# 웹캠 실행
cap = cv2.VideoCapture(0)

# 낮은 해상도 (예: 320x240)로 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
   
    # 프레임 수 조절절
    frame_count += 1

    if frame_count %3 != 0:
        continue

    # 영상 좌우 반전 및 색상 변환
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # 각 손의 제스처와 검지 좌표를 저장할 리스트
    detected_gestures = []
    index_positions = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 손 랜드마크 데이터 추출
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            input_data = np.array(landmarks).flatten().reshape(1, -1)
            prediction = model.predict(input_data)
            gesture = np.argmax(prediction)
            detected_gestures.append(gesture)

            # 검지 손가락 끝 좌표 (8번 포인트) 추출 후 화면 좌표로 변환
            index_finger = hand_landmarks.landmark[8]
            x = int(index_finger.x * screen_width)
            y = int(index_finger.y * screen_height)

            #smoothing 추가
            alpha = 0.8
            prev_x, prev_y = x,y

            smooth_x = int(prev_x + alpha * (x - prev_x))
            smooth_y = int(prev_y + alpha * (y - prev_y))
            prev_x, prev_y = smooth_x, smooth_y

            index_positions.append((smooth_x, smooth_y))

        # 두 손 이상의 제스처를 조합해서 특정 동작을 실행
        if 0 in detected_gestures and 1 in detected_gestures:
            # 한 손이 gesture 0, 다른 손이 gesture 1이면 좌클릭
            pyautogui.click()
        elif 0 in detected_gestures and 2 in detected_gestures:
            # 한 손이 gesture 0, 다른 손이 gesture 2이면 우클릭
            pyautogui.rightClick()
        else:
            # 그 외에는 각 손의 개별 제스처에 따라 동작 실행
            for gesture, pos in zip(detected_gestures, index_positions):
                # 마우스 커서 이동 (여기서는 각 손의 검지 좌표로 이동)
                # pyautogui.moveTo(pos[0], pos[1])

                #윈도우만
                ctypes.windll.user32.SetCursorPos(x, y)
                
                if gesture == 0:
                    # gesture 0: 아무 동작 없음
                    pass
                
                #elif gesture == 1:
                    # gesture 1 단독일 경우에는 좌클릭 동작을 미실행 (조합으로만 실행)
                    #pass
                #elif gesture == 2:
                    # gesture 2 단독일 경우에는 우클릭 동작을 미실행 (조합으로만 실행)
                    #pass   
                elif gesture == 3:
                    pyautogui.press('volumeup')
                elif gesture == 4:
                    pyautogui.press('volumedown')
                elif gesture == 5:
                    pyautogui.press('playpause')
                elif gesture == 6:
                    pyautogui.hotkey('alt', 'left')

    # 화면 출력
    cv2.imshow('Hand Gesture Control', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
