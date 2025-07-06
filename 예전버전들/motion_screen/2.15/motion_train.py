import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking

# CSV 파일 불러오기
csv_filename = "C:/Users/ghksc/Desktop/project/motion_screen/2.15/gesture_data.csv"

df = pd.read_csv(csv_filename)

# 사용 컬럼: sequence_id, frame_index, timestamp, x_0 ~ z_20, label
# 여기서는 feature로 timestamp를 제외한 x,y,z 좌표만 사용합니다.
feature_columns = df.columns[3:-1]  # x_0부터 z_20까지

# sequence_id별로 그룹화하여 시퀀스 데이터와 라벨을 추출
grouped = df.groupby('sequence_id')
sequences = []
labels = []
for seq_id, group in grouped:
    group = group.sort_values('frame_index')
    seq = group[feature_columns].values.astype(np.float32)
    sequences.append(seq)
    # 해당 시퀀스의 라벨은 모든 행에 동일하므로 첫 번째 값 사용
    label = group['label'].iloc[0]
    labels.append(label)

# 각 시퀀스의 길이가 다를 수 있으므로, 가장 긴 시퀀스 길이에 맞춰 패딩 처리 (후쪽에 0 추가)
max_seq_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_seq_length, dtype='float32', padding='post', truncating='post')

# 라벨 인코딩 (문자열 -> 정수) 후 one-hot 인코딩
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_onehot = to_categorical(labels_encoded)

print("시퀀스 개수:", len(padded_sequences))
print("패딩 후 시퀀스 shape:", padded_sequences.shape)
print("라벨 shape:", labels_onehot.shape)

# LSTM 모델 구성
timesteps = padded_sequences.shape[1]  # 시퀀스 길이
features = padded_sequences.shape[2]   # 특징 개수 (예: 21*3 = 63)
num_classes = labels_onehot.shape[1]

model = Sequential([
    # 패딩된 0값을 무시하기 위한 Masking 층
    Masking(mask_value=0., input_shape=(timesteps, features)),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 모델 학습 (예: 50 에포크)
history = model.fit(padded_sequences, labels_onehot, epochs=50, batch_size=8, validation_split=0.2)

# 모델 저장 (예: gesture_model.h5 라는 이름으로 저장)
model.save('gesture_model.h5')

print("gesture_model.h5")

