import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, ReLU
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../../data')

train_data_path = os.path.join(DATA_PATH, 'dataset', 'train')

# 증강
train_datagen = ImageDataGenerator(
    rescale = 1. / 225,
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    horizontal_flip = True,
    brightness_range = [0.9, 1.1]
)

# load data
train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size = (224, 224),
    batch_size = 32,
    class_mode = 'categorical'
)

num_classes = len(train_generator.class_indices)
print(f'총 {num_classes}개의 클래스가 있음 : {train_generator.class_indices}')

# VGG16
use_transfer_learning = True

if use_transfer_learning:
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224, 3))

    # 마지막 4개 레이어만 학습 가능하게 조정
    for layer in base_model.layers[:-4]:
        layer.trainable = False
    for layer in base_model.layers[-4:]:
        layer.trainable = True
    
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

else:
    print('CNN 기본 모델 사용')
    model = Sequential([
        Conv2D(32, (3, 3), padding='same'),
        ReLU(),
        BatchNormalization(),
        MaxPooling2D(2, 2),
    
        Conv2D(64, (3, 3), padding='same'),
        ReLU(),
        BatchNormalization(),
        MaxPooling2D(2, 2),
    
        Conv2D(128, (3, 3), padding='same'),
        ReLU(),
        BatchNormalization(),
        MaxPooling2D(2, 2),
    
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

optimizer = Adam(learning_rate=0.001) 

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks 
lr_reduction = ReduceLROnPlateau(monitor='loss', patience=5, factor=0.5, min_lr=1e-6)

# 학습
history = model.fit(
    train_generator,
    epochs=50,
    callbacks=[lr_reduction]
)

# save model
model_save_path = os.path.join(DATA_PATH, 'cnn_model_fixed.h5')
model.save(model_save_path)
print(f'모델 학습 및 저장 완료 : {model_save_path}')

def plot_training(history):
    acc = history.history['accuracy']
    loss = history.history['loss']
    
    epochs = range(len(acc))
    
    plt.figure(figsize=(10, 4))
    
    # 정확도 
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.show()

plot_training(history)