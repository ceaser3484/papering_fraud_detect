import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../../data')
train_data_path = os.path.join(DATA_PATH, 'dataset', 'train')
augmented_path = os.path.join(DATA_PATH, 'dataset', 'augmented_train') 

# 부족한 클래스
target_counts = {
    "틈새과다": 200,
    "반점": 200,
    "이음부불량": 200,
    "창틀,문틀수정": 200,
    "울음": 200,
    "녹오염": 200,
    "가구수정": 200
}

if not os.path.exists(augmented_path):
    os.makedirs(augmented_path)

# 증강
augmentation = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=(0.7, 1.3),
    brightness_range=(0.5, 1.5),
    horizontal_flip=True,
    channel_shift_range=0.3,
    fill_mode='reflect'
)

# 부족한 클래스 데이터 증강
for class_name, target_count in target_counts.items(): 
    class_folder = os.path.join(train_data_path, class_name)
    augmented_folder = os.path.join(augmented_path, class_name)

    if not os.path.exists(class_folder): 
        print(f"{class_name} 폴더 없음, 건너뜀")
        continue

    if not os.path.exists(augmented_folder):
        os.makedirs(augmented_folder)

    existing_files = os.listdir(class_folder)
    num_existing = len(existing_files)

    if num_existing >= target_count:
        print(f"{class_name} 클래스는 충분한 데이터가 있음 ({num_existing}개), 증강하지 않음")
        continue

    # 증강할 이미지 개수 
    num_to_generate = target_count - num_existing
    print(f"{class_name} 증강 진행: {num_existing}개 → {target_count}개")

    # 증강 실행
    for i in range(num_to_generate):
        img_name = existing_files[i % num_existing]  
        img_path = os.path.join(class_folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)  
        aug_iter = augmentation.flow(img, batch_size=1)

        aug_img = next(aug_iter)[0] 
        aug_img = (aug_img * 255).astype(np.uint8)  
        aug_img = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)  

        new_img_name = f"aug_{i}_{img_name}"
        new_img_path = os.path.join(augmented_folder, new_img_name)
        cv2.imwrite(new_img_path, aug_img)

print("부족한 데이터 증강 완료!")

# 증강 데이터 로드
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=(0.7, 1.3),
    brightness_range=(0.7, 1.4),
    horizontal_flip=True,
    channel_shift_range=0.3,
    fill_mode='reflect'
)

train_generator = train_datagen.flow_from_directory(
    augmented_path, 
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

num_classes = len(train_generator.class_indices)
print(f"총 {num_classes}개의 클래스: {train_generator.class_indices}")

# ResNet50 + Fine-Tuning
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # 처음엔 고정

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])

learning_rate = 2e-4
optimizer = Adam(learning_rate=learning_rate)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

for layer in base_model.layers[-4:]: 
    layer.trainable = True

lr_reduction = ReduceLROnPlateau(monitor="loss", patience=3, factor=0.7, min_lr=1e-6)
early_stopping = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)

history = model.fit(
    train_generator,
    epochs=50,
    callbacks=[lr_reduction, early_stopping]
)

# save
model_save_path = os.path.join(DATA_PATH, "cnn_model_fixed.h5")
model.save(model_save_path)
print(f"모델 학습 및 저장 완료: {model_save_path}")

def plot_training(history):
    acc = history.history["accuracy"]
    loss = history.history["loss"]

    epochs = range(len(acc))

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, "b", label="Training Accuracy")
    plt.title("Training Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, "r", label="Training Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.show()

plot_training(history)