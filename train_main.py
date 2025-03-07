import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../../data")

train_data_path = os.path.join(DATA_PATH, "dataset", "train")

# 데이터 증강
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

# load data
train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

num_classes = len(train_generator.class_indices)
print(f"총 {num_classes}개의 클래스 존재: {train_generator.class_indices}")

# ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False 
# 분류기 추가
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

#  첫 번째 학습
optimizer = Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    epochs=10,  
)

# Fine-Tuning
for layer in base_model.layers[-10:]:  # 마지막 10개 레이어만 학습 
    layer.trainable = True

optimizer_fine = SGD(learning_rate=1e-5, momentum=0.9)
model.compile(
    optimizer=optimizer_fine,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_generator,
    epochs=30, 
)

model_save_path = os.path.join(DATA_PATH, 'cnn_model_finetuned.h5')
model.save(model_save_path)
print(f"Fine-Tuning 완료! 모델 저장됨: {model_save_path}")

def plot_training(history):
    acc = history.history['accuracy']
    loss = history.history['loss']
    
    epochs = range(len(acc))
    
    plt.figure(figsize=(10, 4))
    
    # 정확도 그래프
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    
    # 손실 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.show()

plot_training(history_fine)