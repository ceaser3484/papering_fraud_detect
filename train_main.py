import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Model
from sklearn.utils.class_weight import compute_class_weight
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
    zoom_range=(0.7, 1.3),
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
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# 클래스 수
num_classes = len(train_generator.class_indices)
print(f"총 {num_classes}개의 클래스: {train_generator.class_indices}")

# 클래스 가중치 계산
labels = train_generator.classes
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)
class_weight_dict = dict(zip(np.unique(labels), class_weights))
print("[INFO] 클래스 가중치:", class_weight_dict)

# 모델 구성
input_tensor = Input(shape=(224, 224, 3))
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
output_tensor = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=input_tensor, outputs=output_tensor)
model.trainable = True

# 컴파일
model.compile(
    optimizer=Adam(learning_rate=5e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 콜백
early_stopping = EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-6)

# 학습
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weight_dict 
)

# 모델 저장
model_save_path = os.path.join(DATA_PATH, "cnn_model_weighted.h5")
model.save(model_save_path, save_format="tf")
print(f"[완료] 모델 저장됨: {model_save_path}")

# 시각화
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
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, "r", label="Training Loss")
    plt.plot(epochs, val_loss, "orange", label="Validation Loss")
    plt.title("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training(history)