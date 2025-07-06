import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 데이터 로드
df = pd.read_csv('hand_gestures.csv', header=None)
X = df.iloc[:, 1:].values  # 랜드마크 데이터
y = df.iloc[:, 0].values  # 레이블

# 원-핫 인코딩
from tensorflow.keras.utils import to_categorical
y = to_categorical(y, num_classes=10)  # 0~9까지 10개의 클래스

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 정의
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')  # 10개의 클래스 (0~9)
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# 모델 저장
model.save('hand_gesture_model.h5')
print("🎉 모델이 'hand_gesture_model.h5'로 저장되었습니다.")
