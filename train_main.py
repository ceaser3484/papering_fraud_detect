import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../../data") 

train_data_path = os.path.join(DATA_PATH, "dataset", "train")
val_data_path = os.path.join(DATA_PATH, "dataset", "train") 

# 전처리
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    validation_split = 0.2
)

# load data
train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    subset='validation'
)

# 가중치 계산
class_counts = Counter(train_generator.classes)
class_labels = np.array(list(class_counts.keys()))
class_weights = compute_class_weight(class_weight = 'balanced', classes=class_labels, y=train_generator.classes)
class_weight_dict = dict(zip(class_labels, class_weights))

print('클래스별 데이터 수 : ', class_counts)
print('클래스별 가중치 : ', class_weight_dict, '\n')

use_transfer_learning = True # False -> 기본 CNN 모델 사용

if use_transfer_learning:
    base_model = VGG16(weights = 'imagenet', include_top = False, input_shape = (224, 224, 3))
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation = 'relu'),
        Dropout(0.5),
        Dense(len(train_generator.class_indices), activation = 'softmax')
    ])
    
else:
    print('기본 CNN')
    model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation = 'relu', padding = 'same'),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation = 'relu', padding = 'same'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(128, activation = 'relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation = 'softmax')
])

# compile
model.compile(
    optimizer = Adam(learning_rate = 0.0001),
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

# 학습
model.fit(
    train_generator,
    validation_data = val_generator,
    epochs = 20,
    class_weight = class_weight_dict
)

# model save
model_save_path = os.path.join(DATA_PATH, 'cnn_model_fixed.h5')
model.save(model_save_path)
print('모델 학습 및 저장 완료')

