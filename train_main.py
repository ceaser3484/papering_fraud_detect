import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import GlobalAveragePooling2D

batch_size = 32
train_ratio = 0.8
learning_rate = 0.0001
epochs = 40

train_data_path = '/Users/soohyeon/Desktop/AI_project/open/train'
#test_data_path = '/Users/soohyeon/Desktop/AI_project/open/test'

# 데이터 증강 및 전처리
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,  # 회전 범위를 40 → 20으로 줄임
    width_shift_range=0.2,  # 가로 이동 범위를 0.3 → 0.2로 줄임
    height_shift_range=0.2,  # 세로 이동 범위를 0.3 → 0.2로 줄임
    shear_range=0.15,  # 기울임 강도를 0.3 → 0.15로 줄임
    zoom_range=0.2,  # 확대/축소 범위를 0.3 → 0.2로 줄임
    horizontal_flip=True,  # 좌우 반전 유지
    fill_mode='nearest',
    brightness_range=[0.8, 1.2],  # 밝기 조정을 [0.7, 1.3] → [0.8, 1.2]로 완화
    validation_split=1 - train_ratio
)

# 데이터 로딩
train_generator = datagen.flow_from_directory(
    train_data_path,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    train_data_path,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# ResNet50 기반 모델 생성
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True
for layer in base_model.layers[:120]:  
    layer.trainable = False  


model = Sequential([
    base_model,
    Flatten(),
    Dense(1024, activation="relu", kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(512, activation="relu", kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(len(train_generator.class_indices), activation="softmax")
])

# 모델 컴파일
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# 모델 저장 경로
checkpoint_path = "AI6_project_classification_model.h5"
final_model_path = "AI6_project_model_classification_final.h5"

# 콜백 설정 (모델 저장, 조기 종료, 학습률 감소)
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)

# 모델 학습
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# 최종 모델 저장
model.save(final_model_path)
print(f"최종 모델이 '{final_model_path}'에 저장되었습니다.")

# 검증 데이터 평가
loss, accuracy = model.evaluate(validation_generator)
print(f"검증 데이터에서 모델의 정확도: {accuracy * 100:.2f}%")

# 학습 과정 시각화 함수
def plot_training(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(len(acc))

    plt.figure(figsize=(10, 4))

    # 정확도 그래프
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, "b", label="Training Accuracy")
    plt.plot(epochs_range, val_acc, "g", label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    # 손실 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, "r", label="Training Loss")
    plt.plot(epochs_range, val_loss, "orange", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.show()
