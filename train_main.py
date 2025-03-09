import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../../data')
train_data_path = os.path.join(DATA_PATH, 'dataset', 'train')

# 증강
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=(0.8, 1.2),  
    horizontal_flip=True,
    brightness_range=(0.7, 1.4),  
    fill_mode='reflect'  
)
# load data
train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

num_classes = len(train_generator.class_indices)
print(f'총 {num_classes}개의 클래스: {train_generator.class_indices}')

# fine_tuning
use_transfer_learning = True

if use_transfer_learning:
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # 처음에는 고정

    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),  
        Dropout(0.4), 
        Dense(num_classes, activation='softmax')
    ])
else:
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
        BatchNormalization(),  
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(), 
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),  
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.4), 
        Dense(num_classes, activation='softmax')
    ])

learning_rate = 5e-4  

optimizer = Adam(learning_rate=learning_rate)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-Tuning (20번째 epoch부터 몇 개의 레이어 해제)
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Callback
lr_reduction = ReduceLROnPlateau(monitor='loss', patience=3, factor=0.7, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# 학습
history = model.fit(
    train_generator,
    epochs=50,
    callbacks=[lr_reduction, early_stopping]
)

# 모델 저장
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
    
    # loss 
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.show()
    
plot_training(history)