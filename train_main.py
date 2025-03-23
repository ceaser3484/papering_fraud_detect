import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../../data')
train_data_path = os.path.join(DATA_PATH, 'dataset', 'train')

# 데이터 증강 설정
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=(0.8, 1.2),
    brightness_range=(0.8, 1.2),
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# 데이터 로딩
train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# 클래스 개수
num_classes = len(train_generator.class_indices)
print(f"총 {num_classes}개의 클래스: {train_generator.class_indices}")

# Functional API 모델 정의
input_tensor = Input(shape=(224, 224, 3))
base_model = ResNet50(weights="imagenet", include_top=False, input_tensor=input_tensor)

# 일부 레이어 fine-tuning
for layer in base_model.layers[:-20]:
    layer.trainable = False
for layer in base_model.layers[-20:]:
    layer.trainable = True

# 커스텀 레이어 추가
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output_tensor = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=input_tensor, outputs=output_tensor)

# 컴파일
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 콜백 설정
early_stopping = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-6)

# 학습
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[lr_scheduler, early_stopping]
)

# 모델 저장
model_path = os.path.join(DATA_PATH, 'cnn_model_gradcam.h5')
model.save(model_path)
print(f"모델 저장 완료: {model_path}")

# 학습 시각화
def plot_training(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, "b", label="Training Accuracy")
    plt.plot(epochs, val_acc, "g", label="Validation Accuracy")
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, "r", label="Training Loss")
    plt.plot(epochs, val_loss, "orange", label="Validation Loss")
    plt.legend()
    plt.title("Loss")

    plt.tight_layout()
    plt.show()

plot_training(history)