import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# Mediapipe 손 모듈 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 정적 제스처와 동적 제스처 레이블 번호 입력받기
preset_static = input("원하는 정적 손동작의 레이블 번호를 입력하세요 (예: 0, 1, 2 ...): ")
try:
    static_label = int(preset_static)
except ValueError:
    print("잘못된 입력입니다. 기본 정적 레이블 0으로 설정합니다.")
    static_label = 0

preset_dynamic = input("원하는 동적 손동작의 레이블 번호를 입력하세요 (예: 0, 1, 2 ...): ")
try:
    dynamic_label = int(preset_dynamic)
except ValueError:
    print("잘못된 입력입니다. 기본 동적 레이블 0으로 설정합니다.")
    dynamic_label = 0

# 데이터 저장용 리스트 (정적/동적 각각)
static_data = []    # 각 행: [label, 21*3개 값]
dynamic_data = []   # 각 행: [label, sequence_length, (프레임1 랜드마크... 프레임N 랜드마크)]

# 동적 녹화 관련 변수
dynamic_mode = False
current_dynamic_sequence = []  # 동적 녹화 중에 매 프레임의 랜드마크(플래튼 리스트)를 저장

print("데이터 수집을 시작합니다.")
print("  - 정적 데이터는 매 프레임 저장됩니다.")
print("  - 동적 데이터를 녹화하려면 'd'를 누르고, 종료하려면 'f'를 누르세요.")
print("  - 종료는 'q' 키를 누르세요.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 좌우 반전 및 색상 변환
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # 오른손만 인식하도록 (multi_handedness 활용)
    right_hand_found = False
    right_hand_landmarks = None
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # handedness.classification[0].label 는 "Left" 또는 "Right"
            if handedness.classification[0].label == "Right":
                right_hand_found = True
                right_hand_landmarks = hand_landmarks
                break

    if right_hand_found and right_hand_landmarks is not None:
        # 오른손 랜드마크 그리기
        mp_drawing.draw_landmarks(frame, right_hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 21개 랜드마크의 정규화 좌표 추출
        landmarks = []
        for lm in right_hand_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])
        flattened_landmarks = np.array(landmarks).flatten().tolist()  # 길이 63 리스트

        # 동적 녹화 중이면 현재 시퀀스에 추가, 아니면 정적 데이터로 저장
        if dynamic_mode:
            current_dynamic_sequence.append(flattened_landmarks)
            cv2.putText(frame, "Dynamic Mode Recording...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            row = [static_label] + flattened_landmarks
            static_data.append(row)
            cv2.putText(frame, f"Static Label: {static_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Collect Gesture Data', frame)
    key = cv2.waitKey(1) & 0xFF

    # 키 입력 처리
    if key == ord('q'):
        break
    elif key == ord('d'):
        if not dynamic_mode:
            dynamic_mode = True
            current_dynamic_sequence = []  # 새 동적 시퀀스 초기화
            print("동적 녹화 시작됨...")
    elif key == ord('f'):
        if dynamic_mode:
            dynamic_mode = False
            if len(current_dynamic_sequence) > 0:
                seq_length = len(current_dynamic_sequence)
                # 동적 시퀀스는 여러 프레임의 플래튼 리스트를 순서대로 이어붙임
                flattened_sequence = []
                for frame_landmarks in current_dynamic_sequence:
                    flattened_sequence.extend(frame_landmarks)
                row = [dynamic_label, seq_length] + flattened_sequence
                dynamic_data.append(row)
                print(f"동적 녹화 종료됨. {seq_length} 프레임 기록됨.")
            else:
                print("동적 녹화 종료됨. 기록된 데이터 없음.")

cap.release()
cv2.destroyAllWindows()

# CSV 파일로 데이터 추가 저장 (파일이 있으면 추가, 없으면 새로 생성)
static_csv = 'hand_gestures_static.csv'
dynamic_csv = 'hand_gestures_dynamic.csv'

if len(static_data) > 0:
    df_static = pd.DataFrame(static_data)
    df_static.to_csv(static_csv, mode='a', index=False, header=False)
    print(f"정적 손동작 데이터가 '{static_csv}'에 추가 저장되었습니다.")
else:
    print("저장할 정적 데이터가 없습니다.")

if len(dynamic_data) > 0:
    df_dynamic = pd.DataFrame(dynamic_data)
    df_dynamic.to_csv(dynamic_csv, mode='a', index=False, header=False)
    print(f"동적 손동작 데이터가 '{dynamic_csv}'에 추가 저장되었습니다.")
else:
    print("저장할 동적 데이터가 없습니다.")
