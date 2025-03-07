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
    rescale = 1./255,
    rotation_range = 20,  
    width_shift_range = 0.2,  
    height_shift_range = 0.2,  
    shear_range = 0.2,  
    zoom_range = 0.2, 
    horizontal_flip = True,
    brightness_range = [0.8, 1.2], 
    fill_mode='nearest'
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

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Feature Extractor 역할

model = Sequential([
    base_model,
    GlobalAveragePooling2D(), 
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

optimizer = Adam(learning_rate=0.0001)  

# compile
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# callback
lr_reduction = ReduceLROnPlateau(monitor='loss', patience=3, factor=0.5, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# 학습
history = model.fit(
    train_generator,
    epochs=50,
    callbacks=[lr_reduction, early_stopping]
)

# save
model_save_path = os.path.join(DATA_PATH, 'cnn_model_fixed.h5')
model.save(model_save_path)
print(f'모델 학습 및 저장 완료: {model_save_path}')

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

    # 손실값
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.show()

plot_training(history)