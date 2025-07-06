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

print("데이터 수집 모드를 선택하세요:")
print("1: 정적 제스처 수집")
print("2: 동적 제스처 수집")
mode = input("선택 (1 또는 2): ")

if mode == '1':
    # ----- 정적 제스처 수집 -----
    preset = input("정적 제스처의 레이블 번호를 입력하세요 (예: 0, 1, 2 ...): ")
    try:
        static_label = int(preset)
    except ValueError:
        print("잘못된 입력입니다. 기본 레이블 0으로 설정합니다.")
        static_label = 0

    static_data = []  # 정적 데이터 저장 리스트

    print("정적 제스처 데이터 수집을 시작합니다. (종료하려면 'q'를 누르세요)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # 오른손만 인식 (multi_handedness 사용)
        right_hand_found = False
        right_hand_landmarks = None
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                if handedness.classification[0].label == "Right":
                    right_hand_found = True
                    right_hand_landmarks = hand_landmarks
                    break

        if right_hand_found and right_hand_landmarks is not None:
            mp_drawing.draw_landmarks(frame, right_hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # 21개 랜드마크의 정규화 좌표 추출
            landmarks = []
            for lm in right_hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            flattened_landmarks = np.array(landmarks).flatten().tolist()  # 길이 63 리스트

            # 정적 데이터 한 행: [레이블, 63개 값]
            row = [static_label] + flattened_landmarks
            static_data.append(row)

            cv2.putText(frame, f"Static Label: {static_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Static Gesture Data Collection', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # CSV 파일로 저장 (추가 저장 모드)
    static_csv = 'hand_gestures_static.csv'
    if len(static_data) > 0:
        df_static = pd.DataFrame(static_data)
        df_static.to_csv(static_csv, mode='a', index=False, header=False)
        print(f"정적 제스처 데이터가 '{static_csv}'에 추가 저장되었습니다.")
    else:
        print("저장할 정적 데이터가 없습니다.")

elif mode == '2':
    # ----- 동적 제스처 수집 -----
    preset = input("동적 제스처의 레이블 번호를 입력하세요 (예: 0, 1, 2 ...): ")
    try:
        dynamic_label = int(preset)
    except ValueError:
        print("잘못된 입력입니다. 기본 레이블 0으로 설정합니다.")
        dynamic_label = 0

    dynamic_data = []       # 동적 데이터 저장 리스트
    current_dynamic_sequence = []  # 현재 녹화 중인 시퀀스
    dynamic_mode = False

    print("동적 제스처 데이터 수집 모드입니다.")
    print("  - 동적 녹화를 시작하려면 'd' 키를 누르세요.")
    print("  - 동적 녹화를 종료하려면 'f' 키를 누르세요.")
    print("  - 종료는 'q' 키를 누르세요.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # 오른손만 인식
        right_hand_found = False
        right_hand_landmarks = None
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                if handedness.classification[0].label == "Right":
                    right_hand_found = True
                    right_hand_landmarks = hand_landmarks
                    break

        if right_hand_found and right_hand_landmarks is not None:
            mp_drawing.draw_landmarks(frame, right_hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # 21개 랜드마크의 정규화 좌표 추출
            landmarks = []
            for lm in right_hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            flattened_landmarks = np.array(landmarks).flatten().tolist()  # 길이 63 리스트

            if dynamic_mode:
                current_dynamic_sequence.append(flattened_landmarks)
                cv2.putText(frame, "Recording Dynamic Sequence...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"Dynamic Label: {dynamic_label}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Dynamic Gesture Data Collection', frame)
        key = cv2.waitKey(1) & 0xFF

        # 키 입력 처리
        if key == ord('q'):
            # 만약 동적 녹화 중이었다면 시퀀스 저장
            if dynamic_mode and len(current_dynamic_sequence) > 0:
                seq_length = len(current_dynamic_sequence)
                flattened_sequence = []
                for frame_landmarks in current_dynamic_sequence:
                    flattened_sequence.extend(frame_landmarks)
                row = [dynamic_label, seq_length] + flattened_sequence
                dynamic_data.append(row)
                print(f"동적 시퀀스 종료됨. {seq_length} 프레임 기록됨.")
            break
        elif key == ord('d'):
            if not dynamic_mode:
                dynamic_mode = True
                current_dynamic_sequence = []  # 새 시퀀스 초기화
                print("동적 녹화 시작됨...")
        elif key == ord('f'):
            if dynamic_mode:
                dynamic_mode = False
                if len(current_dynamic_sequence) > 0:
                    seq_length = len(current_dynamic_sequence)
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

    # CSV 파일로 저장 (추가 저장 모드)
    dynamic_csv = 'hand_gestures_dynamic.csv'
    if len(dynamic_data) > 0:
        df_dynamic = pd.DataFrame(dynamic_data)
        df_dynamic.to_csv(dynamic_csv, mode='a', index=False, header=False)
        print(f"동적 제스처 데이터가 '{dynamic_csv}'에 추가 저장되었습니다.")
    else:
        print("저장할 동적 데이터가 없습니다.")

else:
    print("잘못된 선택입니다. 프로그램을 종료합니다.")
    cap.release()
    cv2.destroyAllWindows()
