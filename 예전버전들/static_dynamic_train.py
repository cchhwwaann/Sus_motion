import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Masking, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

def train_static_model(static_csv='hand_gestures_static.csv'):
    # 정적 데이터 로드 (CSV 파일에는 헤더가 없다고 가정)
    print("정적 제스처 데이터를 불러오는 중...")
    df = pd.read_csv(static_csv, header=None)
    data = df.values
    # 첫 번째 컬럼은 레이블, 나머지는 63개 랜드마크 값
    X = data[:, 1:].astype('float32')
    y = data[:, 0].astype('int')
    
    num_classes = len(np.unique(y))
    y_cat = to_categorical(y, num_classes=num_classes)
    
    # 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)
    
    input_dim = X.shape[1]
    print(f"정적 데이터: {X.shape[0]} 샘플, 입력 차원: {input_dim}, 클래스 수: {num_classes}")
    
    # 간단한 완전 연결(FC) 신경망 모델 구성
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("정적 제스처 모델 학습 시작...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2,
              callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
    
    loss, acc = model.evaluate(X_test, y_test)
    print(f"정적 제스처 모델 정확도: {acc:.4f}")
    
    model.save('hand_gesture_model_static.h5')
    print("정적 제스처 모델이 'hand_gesture_model_static.h5'로 저장되었습니다.")

def train_dynamic_model(dynamic_csv='hand_gestures_dynamic.csv'):
    # 동적 데이터는 각 행마다 시퀀스 길이가 다를 수 있으므로, 직접 파싱합니다.
    print("동적 제스처 데이터를 불러오는 중...")
    dynamic_sequences = []
    labels = []
    
    with open(dynamic_csv, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) < 3:
            continue
        try:
            label = int(parts[0])
            seq_length = int(parts[1])
        except ValueError:
            continue
        # 빈 문자열을 제외하고 float으로 변환
        flattened = list(map(float, filter(None, parts[2:])))
        # 각 프레임은 63개의 값이어야 함
        if len(flattened) != seq_length * 63:
            print("Warning: 시퀀스 길이와 실제 데이터 길이가 일치하지 않습니다. 스킵합니다.")
            continue
        # (seq_length, 63) 형태로 재구성
        sequence = np.array(flattened).reshape(seq_length, 63)
        dynamic_sequences.append(sequence)
        labels.append(label)
    
    if len(dynamic_sequences) == 0:
        print("불러올 동적 데이터가 없습니다.")
        return
    
    # 시퀀스 길이를 모두 동일하게 맞추기 위해 패딩
    max_seq_len = max(seq.shape[0] for seq in dynamic_sequences)
    print(f"동적 데이터: {len(dynamic_sequences)} 샘플, 최대 시퀀스 길이: {max_seq_len}")
    
    padded_sequences = pad_sequences(dynamic_sequences, maxlen=max_seq_len, dtype='float32',
                                       padding='post', truncating='post')
    X = np.array(padded_sequences)  # shape: (num_samples, max_seq_len, 63)
    y = np.array(labels)
    
    num_classes = len(np.unique(y))
    y_cat = to_categorical(y, num_classes=num_classes)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)
    
    # LSTM 기반 모델 구성 (패딩된 부분은 Masking 레이어로 무시)
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(max_seq_len, 63)))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("동적 제스처 모델 학습 시작...")
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2,
              callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
    
    loss, acc = model.evaluate(X_test, y_test)
    print(f"동적 제스처 모델 정확도: {acc:.4f}")
    
    model.save('hand_gesture_model_dynamic.h5')
    print("동적 제스처 모델이 'hand_gesture_model_dynamic.h5'로 저장되었습니다.")

if __name__ == "__main__":
    print("어떤 모델을 학습시키겠습니까?")
    print("1. 정적 제스처 모델 (단일 프레임 기반)")
    print("2. 동적 제스처 모델 (시퀀스 기반)")
    choice = input("선택 (1 또는 2): ")
    if choice == '1':
        train_static_model()
    elif choice == '2':
        train_dynamic_model()
    else:
        print("잘못된 선택입니다.")
