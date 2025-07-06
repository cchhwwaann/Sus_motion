import cv2
import mediapipe as mp
import numpy as np
import time

# --- 1. 초기 설정 및 전역 변수 ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Hand Tracking 모델 설정
hands = mp_hands.Hands(
    model_complexity=0, # 빠르고 가벼운 모델
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# 웹캠 설정
cap = cv2.VideoCapture(0) # 0번 카메라 (기본 웹캠)
if not cap.isOpened():
    print("Error: Could not open video stream. Check your webcam or camera index.")
    exit()

cap_width, cap_height = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

# --- 2. 헬퍼 함수 정의 ---

def calculate_distance(point1, point2):
    """두 랜드마크 포인트 사이의 유클리드 거리를 계산합니다."""
    return np.linalg.norm(np.array(point1) - np.array(point2))

def get_landmark_coords(landmark, img_width, img_height):
    """랜드마크 객체에서 픽셀 좌표를 추출합니다."""
    return int(landmark.x * img_width), int(landmark.y * img_height)

# --- 3. 메인 루프 ---
print("--- 검지-중지 손가락 최대 거리 측정 시작 ---")
print("카메라 앞에서 검지와 중지를 최대한 벌려보세요.")
print("측정된 거리가 콘솔에 계속 출력됩니다. 최대값을 확인하세요.")

max_observed_distance = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) # 거울 모드
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 손 랜드마크 그리기
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # 검지 끝과 중지 끝 랜드마크 좌표 추출
            index_tip = get_landmark_coords(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP], cap_width, cap_height)
            middle_tip = get_landmark_coords(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP], cap_width, cap_height)

            # 거리 계산
            current_distance = calculate_distance(index_tip, middle_tip)

            # 최대 거리 업데이트
            if current_distance > max_observed_distance:
                max_observed_distance = current_distance
            
            # 현재 거리 및 최대 거리 출력
            print(f"Current Distance: {int(current_distance)} px, Max Observed: {int(max_observed_distance)} px")

            # 화면에 거리 표시 (선택 사항)
            cv2.putText(frame, f'Dist: {int(current_distance)}px', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Max: {int(max_observed_distance)}px', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)


    cv2.imshow('Finger Distance Calibration', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 4. 종료 처리 ---
cap.release()
cv2.destroyAllWindows()
hands.close()
print("--- 측정 종료 ---")
print(f"최종 최대 거리: {int(max_observed_distance)} px")