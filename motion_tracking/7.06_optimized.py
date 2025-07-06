import cv2
import mediapipe as mp
import numpy as np
from pynput.mouse import Controller, Button
import time
import math
import tkinter as tk # Tkinter 라이브러리 추가

# --- 설정값 (조정 가능) ---
# 화면 해상도를 프로그램 실행 시 자동으로 가져옵니다.
root = tk.Tk()
root.withdraw() # Tkinter 윈도우를 숨깁니다.
SCREEN_WIDTH = root.winfo_screenwidth()
SCREEN_HEIGHT = root.winfo_screenheight()
root.destroy() # Tkinter 루트 윈도우를 파괴하여 리소스 해제

print(f"Detected Screen Resolution: {SCREEN_WIDTH}x{SCREEN_HEIGHT}") # 감지된 해상도 확인용 출력

# 손 인식 관심 영역 (ROI) 설정
HAND_ROI_WIDTH = 320      
HAND_ROI_HEIGHT = 240 

# 커서 이동 감도
CURSOR_SENSITIVITY_X = 2.0
CURSOR_SENSITIVITY_Y = 2.0

# --- 스무딩 관련 설정 ---
ALPHA_SMOOTHING = 0.25 # 0.0에 가까울수록 부드러움, 1.0에 가까울수록 민감함

# --- 커서 기준점 오프셋 설정 ---
# 손바닥 중앙의 X좌표를 당기는 값 (픽셀 단위)
# 양수: 왼쪽으로 당김, 음수: 오른쪽으로 당김 (화면 기준)
CURSOR_X_OFFSET_FROM_CENTER = 100 # 사용자 조정값: 100픽셀

# --- 화면 경계 여백(Padding) 설정 ---
# 마우스 커서가 화면 가장자리에서 얼마나 안쪽까지 움직일지 결정 (픽셀)
SCREEN_EDGE_PADDING_X = 20 # 좌우 여백 (예: 20픽셀)
SCREEN_EDGE_PADDING_Y = 20 # 상하 여백 (예: 20픽셀)


# --- 클릭 관련 설정 ---
# **기본 자세 (엄지, 검지, 중지 모두 떨어져 있음) 기준**
# 이 값보다 엄지-검지 거리가 멀면 '떨어져있음'으로 간주 (기본 자세 확인)
THUMB_INDEX_SEPARATE_THRESHOLD_BASE = 0.08 
# 이 값보다 엄지-중지 거리가 멀면 '떨어져있음'으로 간주 (기본 자세 확인)
THUMB_MIDDLE_SEPARATE_THRESHOLD_BASE = 0.08 

# **좌클릭 (엄지-검지 붙어있는 동안 유지) 기준**
# 이 값보다 엄지-검지 거리가 가까우면 '붙어있음'으로 간주 (클릭 시작)
THUMB_INDEX_CLICK_CONTACT_THRESHOLD = 0.04 
# 이 값보다 엄지-검지 거리가 멀면 '떨어져있음'으로 간주 (클릭 종료)
THUMB_INDEX_CLICK_RELEASE_THRESHOLD = 0.08 

# **우클릭 (중지-엄지 0.5초 이상 붙였다 뗄 때) 기준**
# 이 값보다 엄지-중지 거리가 가까우면 '붙어있음'으로 간주 (우클릭 탭 시작)
THUMB_MIDDLE_RIGHT_CLICK_CONTACT_THRESHOLD = 0.04 
# 이 값보다 엄지-중지 거리가 멀면 '떨어져있음'으로 간주 (우클릭 탭 완료)
THUMB_MIDDLE_RIGHT_CLICK_RELEASE_THRESHOLD = 0.08 
RIGHT_CLICK_HOLD_TIME = 0.5 # 우클릭을 위해 중지-엄지 접촉을 유지해야 하는 시간

# **더블클릭 (엄지-검지-중지 모두 닿을 때) 기준**
# 이 값보다 엄지-검지 거리가 가까우면 '붙어있음'으로 간주
TRIPLE_FINGER_CONTACT_THRESHOLD = 0.04 

# 모든 클릭 이벤트(좌, 우, 더블) 후 다음 클릭까지의 최소 대기 시간 (초)
GLOBAL_CLICK_COOLDOWN_TIME = 0.3 

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 마우스 컨트롤러 초기화
mouse = Controller()

# --- 클릭 상태 관리 변수 ---
# 마지막으로 마우스 클릭 이벤트가 발생한 시간 (모든 클릭 공통 쿨다운용)
prev_any_click_event_time = 0 

# **좌클릭 (엄지-검지) 관련 상태**
is_left_button_down = False # 마우스 좌클릭 버튼이 현재 눌린 상태인지 (클릭 유지)
is_thumb_index_contact_for_hold = False # 엄지-검지가 붙어있어 클릭 유지 중인지
last_left_click_release_time = 0 # 마지막 좌클릭(단일)이 해제된 시간 (더블클릭은 이제 별도 제스처)

# **우클릭 (중지-엄지) 관련 상태**
is_middle_contact_for_right_click_tap = False # 우클릭을 위해 중지가 엄지에 붙어있는지 (탭 시작)
right_click_contact_start_time = 0 # 우클릭 탭이 시작된 시간
right_click_triggered_in_session = False # 현재 우클릭 탭 세션에서 이미 우클릭 발생했는지 여부

# **더블클릭 (엄지-검지-중지 모두 닿을 때) 관련 상태**
is_triple_finger_contact = False # 세 손가락이 모두 닿아있는지 여부
double_click_triggered_in_session = False # 현재 세 손가락 접촉 세션에서 이미 더블클릭 발생했는지 여부

# 현재 손이 '기본 자세'를 유지하고 있는지 여부
is_base_pose_active = False 

# 두 랜드마크 사이의 유클리드 거리 계산 함수 (정규화된 좌표 기준)
def calculate_distance(landmark1, landmark2):
    return math.sqrt((landmark1.x - landmark2.x)**2 + 
                     (landmark1.y - landmark2.y)**2 + 
                     (landmark1.z - landmark2.z)**2) 

# 웹캠 초기화
cap = cv2.VideoCapture(0) 

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 카메라 해상도 변수를 여기서 정의 (cap.set() 이후)
CAMERA_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
CAMERA_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# HAND_ORIGIN_X에 CURSOR_X_OFFSET_FROM_CENTER를 적용
HAND_ORIGIN_X_ADJUSTED = (CAMERA_WIDTH // 2) + CURSOR_X_OFFSET_FROM_CENTER
HAND_ORIGIN_Y_ADJUSTED = (CAMERA_HEIGHT // 2) # Y좌표는 그대로 중앙 유지

smooth_cursor_x, smooth_cursor_y = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 

with mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    print("손 인식 GUI 제어 프로그램 시작. 'q'를 눌러 종료.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("카메라를 읽을 수 없습니다.")
            continue

        # 이미지 좌우 반전 (거울 모드)
        image = cv2.flip(image, 1)

        # BGR을 RGB로 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 손 인식 처리
        results = hands.process(image_rgb)

        # 현재 시간 기록
        current_time = time.time()

        # 이미지에 랜드마크 그리기
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2), 
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)) 

                # --- 1. 마우스 커서 이동 로직 (손바닥 중앙 - MIDDLE_FINGER_MCP 기준) ---
                cursor_base_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP] 
                
                cursor_base_x_cam = int(cursor_base_landmark.x * CAMERA_WIDTH)
                cursor_base_y_cam = int(cursor_base_landmark.y * CAMERA_HEIGHT)

                # 손바닥 중앙 위치를 화면에 표시
                cv2.circle(image, (cursor_base_x_cam, cursor_base_y_cam), 8, (255, 0, 0), -1) 

                # relative_hand_x 계산 시 HAND_ORIGIN_X_ADJUSTED 사용
                relative_hand_x = cursor_base_x_cam - HAND_ORIGIN_X_ADJUSTED
                relative_hand_y = cursor_base_y_cam - HAND_ORIGIN_Y_ADJUSTED

                target_cursor_x = int(relative_hand_x * (SCREEN_WIDTH / HAND_ROI_WIDTH) * CURSOR_SENSITIVITY_X + SCREEN_WIDTH / 2)
                target_cursor_y = int(relative_hand_y * (SCREEN_HEIGHT / HAND_ROI_HEIGHT) * CURSOR_SENSITIVITY_Y + SCREEN_HEIGHT / 2)
                
                # target_cursor_x와 target_cursor_y에 일차적으로 클립 적용
                target_cursor_x = np.clip(target_cursor_x, 0, SCREEN_WIDTH - 1)
                target_cursor_y = np.clip(target_cursor_y, 0, SCREEN_HEIGHT - 1)

                smooth_cursor_x = ALPHA_SMOOTHING * target_cursor_x + (1 - ALPHA_SMOOTHING) * smooth_cursor_x
                smooth_cursor_y = ALPHA_SMOOTHING * target_cursor_y + (1 - ALPHA_SMOOTHING) * smooth_cursor_y

                # --- 최종 커서 좌표에 np.clip 적용 및 패딩 추가 ---
                # min_x, max_x, min_y, max_y를 패딩을 고려하여 설정
                min_x_clip = SCREEN_EDGE_PADDING_X
                max_x_clip = SCREEN_WIDTH - 1 - SCREEN_EDGE_PADDING_X
                min_y_clip = SCREEN_EDGE_PADDING_Y
                max_y_clip = SCREEN_HEIGHT - 1 - SCREEN_EDGE_PADDING_Y

                final_cursor_x = int(np.clip(smooth_cursor_x, min_x_clip, max_x_clip))
                final_cursor_y = int(np.clip(smooth_cursor_y, min_y_clip, max_y_clip))

                mouse.position = (final_cursor_x, final_cursor_y)

                cv2.putText(image, f"Cursor: ({final_cursor_x}, {final_cursor_y})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.circle(image, (cursor_base_x_cam, cursor_base_y_cam), 5, (255, 0, 0), -1)

                # --- 2. 클릭 로직 ---

                # 랜드마크 거리 계산
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                thumb_index_distance = calculate_distance(index_finger_tip, thumb_tip)
                thumb_middle_distance = calculate_distance(middle_finger_tip, thumb_tip)
                
                # 세 손가락 모두 닿았는지 확인 (더블클릭용)
                is_all_three_fingers_contact = \
                    (thumb_index_distance < TRIPLE_FINGER_CONTACT_THRESHOLD) and \
                    (thumb_middle_distance < TRIPLE_FINGER_CONTACT_THRESHOLD) and \
                    (calculate_distance(index_finger_tip, middle_finger_tip) < TRIPLE_FINGER_CONTACT_THRESHOLD * 1.5) 

                cv2.putText(image, f"Idx-Th Dist: {thumb_index_distance:.3f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(image, f"Mid-Th Dist: {thumb_middle_distance:.3f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(image, f"3-Fingers: {is_all_three_fingers_contact}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # --- 기본 자세 (엄지, 검지, 중지 모두 떨어져 있음) 감지 ---
                is_base_pose_active = \
                    (thumb_index_distance > THUMB_INDEX_SEPARATE_THRESHOLD_BASE) and \
                    (thumb_middle_distance > THUMB_MIDDLE_SEPARATE_THRESHOLD_BASE) and \
                    (calculate_distance(index_finger_tip, middle_finger_tip) > THUMB_MIDDLE_SEPARATE_THRESHOLD_BASE) 
                
                # --- 더블클릭 (엄지-검지-중지 모두 닿을 때) 로직 처리 ---
                if is_all_three_fingers_contact and not is_triple_finger_contact:
                    is_triple_finger_contact = True # 세 손가락 닿음 상태 시작

                    # 모든 클릭 쿨다운 중이 아닐 때만 더블클릭 트리거
                    if (current_time - prev_any_click_event_time > GLOBAL_CLICK_COOLDOWN_TIME) and \
                       not is_left_button_down and not is_middle_contact_for_right_click_tap: 
                        
                        mouse.click(Button.left, 2) # 더블클릭 실행
                        print(f"더블클릭 감지! (세 손가락 모두 닿음)")
                        cv2.putText(image, "DOUBLE CLICK!", (cursor_base_x_cam + 20, cursor_base_y_cam - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                        prev_any_click_event_time = current_time # 모든 클릭 공통 쿨다운 갱신
                        double_click_triggered_in_session = True # 이 세션에서 더블클릭 발생 플래그 설정
                elif not is_all_three_fingers_contact and is_triple_finger_contact:
                    is_triple_finger_contact = False # 세 손가락 떨어짐 상태 해제
                    double_click_triggered_in_session = False # 다음 더블클릭 시도를 위해 플래그 초기화


                # --- 좌클릭 (엄지-검지 붙어있는 동안 유지) 로직 처리 ---
                current_thumb_index_contact = (thumb_index_distance < THUMB_INDEX_CLICK_CONTACT_THRESHOLD)
                current_thumb_index_released = (thumb_index_distance > THUMB_INDEX_CLICK_RELEASE_THRESHOLD)

                # 엄지-검지 붙음 감지 (클릭 시작)
                # 더블클릭 동작 중이 아닐 때만 좌클릭 시작
                if current_thumb_index_contact and not is_thumb_index_contact_for_hold and not is_triple_finger_contact:
                    # 기본 자세가 아니거나, 우클릭을 시도 중이 아닐 때만 좌클릭 시작
                    # 그리고 모든 클릭 공통 쿨다운이 끝났을 때
                    if (not is_base_pose_active or is_middle_contact_for_right_click_tap == False) and \
                       (current_time - prev_any_click_event_time > GLOBAL_CLICK_COOLDOWN_TIME):
                        
                        mouse.press(Button.left) # 마우스 버튼 누름
                        print(f"좌클릭 시작! (검지-엄지 거리: {thumb_index_distance:.3f})")
                        cv2.putText(image, "CLICK_DOWN!", (cursor_base_x_cam + 20, cursor_base_y_cam - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                        is_left_button_down = True # 버튼이 눌린 상태로 표시
                        prev_any_click_event_time = current_time # 클릭 시작 시 쿨다운 갱신 (누르는 동안 다른 클릭 방지)
                    
                    is_thumb_index_contact_for_hold = True # 엄지-검지 붙음 상태 시작

                # 엄지-검지 떼어짐 감지 (클릭 종료)
                elif current_thumb_index_released and is_thumb_index_contact_for_hold:
                    is_thumb_index_contact_for_hold = False # 떼어졌으므로 붙음 상태 해제
                    
                    if is_left_button_down: # 이전에 버튼이 눌린 상태였다면
                        mouse.release(Button.left) # 마우스 버튼 해제
                        print(f"좌클릭 종료! (검지-엄지 거리: {thumb_index_distance:.3f})")
                        cv2.putText(image, "CLICK_UP!", (cursor_base_x_cam + 20, cursor_base_y_cam - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                        is_left_button_down = False # 버튼 상태 해제


                # --- 우클릭 (중지-엄지 0.5초 이상 붙였다 떼어질 때) 로직 처리 ---
                current_thumb_middle_contact = (thumb_middle_distance < THUMB_MIDDLE_RIGHT_CLICK_CONTACT_THRESHOLD)
                current_thumb_middle_released = (thumb_middle_distance > THUMB_MIDDLE_RIGHT_CLICK_RELEASE_THRESHOLD)

                # 중지-엄지 붙음 감지 (우클릭 시작 준비)
                # 더블클릭 동작 중이 아닐 때만 우클릭 시작
                if current_thumb_middle_contact and not is_middle_contact_for_right_click_tap and not is_triple_finger_contact:
                    is_middle_contact_for_right_click_tap = True
                    right_click_contact_start_time = current_time # 접촉 시작 시간 기록
                    right_click_triggered_in_session = False # 새 탭 세션이므로 우클릭 플래그 초기화
                
                # 중지-엄지 떼어짐 감지 (우클릭 발생)
                elif current_thumb_middle_released and is_middle_contact_for_right_click_tap:
                    is_middle_contact_for_right_click_tap = False # 떼어졌으므로 붙음 상태 해제 (우클릭 완료)

                    # 검지가 엄지에서 떨어져있는 상태여야 하고 (좌클릭 동작과 충돌 방지),
                    # 모든 클릭 공통 쿨다운이 끝났을 때,
                    # 그리고 0.5초 이상 유지했어야 함.
                    if (thumb_index_distance > THUMB_INDEX_CLICK_RELEASE_THRESHOLD) and \
                       (current_time - prev_any_click_event_time > GLOBAL_CLICK_COOLDOWN_TIME) and \
                       (current_time - right_click_contact_start_time >= RIGHT_CLICK_HOLD_TIME) and \
                       not right_click_triggered_in_session: # 이 세션에서 아직 우클릭이 발생하지 않았을 때만
                        
                        mouse.click(Button.right) 
                        print(f"우클릭 감지! (중지-엄지 거리: {thumb_middle_distance:.3f}, 유지: {current_time - right_click_contact_start_time:.3f}s)")
                        cv2.putText(image, "RIGHT CLICK!", (cursor_base_x_cam + 20, cursor_base_y_cam - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                        prev_any_click_event_time = current_time # 모든 클릭 공통 쿨다운 갱신 
                        right_click_triggered_in_session = True # 이 세션에서 우클릭 발생 플래그 설정

        else: # 손이 감지되지 않는 경우, 모든 클릭 관련 상태 초기화
            is_base_pose_active = False
            is_thumb_index_contact_for_hold = False
            if is_left_button_down: # 손이 사라지면 눌린 버튼 해제
                mouse.release(Button.left)
                is_left_button_down = False
            last_left_click_release_time = 0 # 초기화
            is_middle_contact_for_right_click_tap = False
            right_click_contact_start_time = 0 # 초기화
            right_click_triggered_in_session = False
            is_triple_finger_contact = False
            double_click_triggered_in_session = False
            # prev_any_click_event_time은 유지 (쿨다운 추적)

        # 결과 이미지 표시
        cv2.imshow('Hand Gesture Control', image)

        # 'q' 키를 눌러 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("프로그램 종료.")