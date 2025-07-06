import cv2
import mediapipe as mp
import csv
import time

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# CSV 파일명 및 헤더 설정 (각 행: sequence_id, frame_index, timestamp, 21개 랜드마크의 x,y,z, label)
csv_filename = "gesture_data.csv"
header = ["sequence_id", "frame_index", "timestamp"]
for i in range(21):
    header.extend([f"x_{i}", f"y_{i}", f"z_{i}"])
header.append("label")

# 웹캠 열기
cap = cv2.VideoCapture(0)

sequence_id = 0      # 제스처 시퀀스 번호
recording = False    # 녹화 중인지 여부
frame_index = 0      # 시퀀스 내 프레임 번호
sequence_data = []   # 현재 시퀀스의 데이터 저장 리스트

# 수집할 프레임 수 (여기서는 30프레임)
FRAMES_TO_RECORD = 30

print("녹화를 시작하려면 'r', 프로그램 종료는 'q'를 누르세요.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 영상 좌우 반전 및 색상 변환
    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            # 화면에 랜드마크 그리기
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            if recording:
                # 각 프레임의 데이터를 수집 (타임스탬프와 21개 랜드마크의 x,y,z 좌표)
                row = [sequence_id, frame_index, time.time()]
                for lm in handLms.landmark:
                    row.extend([lm.x, lm.y, lm.z])
                row.append("")  # 나중에 라벨을 채워 넣을 자리
                sequence_data.append(row)
                frame_index += 1

                # 30프레임이 채워지면 자동 녹화 종료 및 CSV 저장
                if frame_index >= FRAMES_TO_RECORD:
                    print(f"Sequence {sequence_id} 녹화 종료 (30프레임 수집 완료)")
                    label = input(f"Sequence {sequence_id}의 라벨을 입력하세요: ")
                    for r in sequence_data:
                        r[-1] = label
                    with open(csv_filename, mode='a', newline='') as f:
                        csv_writer = csv.writer(f)
                        # 파일이 비어있으면 헤더 먼저 기록
                        if f.tell() == 0:
                            csv_writer.writerow(header)
                        csv_writer.writerows(sequence_data)
                    sequence_id += 1
                    recording = False
                    frame_index = 0
                    sequence_data = []
    else:
        # 손이 감지되지 않으면 별도 처리(필요시)
        pass

    cv2.imshow("Gesture Recording", frame)
    key = cv2.waitKey(1)

    if key == ord('r') and not recording:
        # 녹화 시작
        print(f"Sequence {sequence_id} 녹화 시작")
        recording = True
        frame_index = 0
        sequence_data = []
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
