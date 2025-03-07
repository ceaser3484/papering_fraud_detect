import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing import image

# 경로
train_data_path = '/Users/soohyeon/Desktop/AI6_WorldAIProject/projectfile/train'  # train 데이터가 있는 폴더 경로
test_data_path = '/Users/soohyeon/Desktop/AI6_WorldAIProject/projectfile/test'  # test 데이터가 있는 폴더 경로

# 데이터 증강
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

# 데이터 로드
train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
# ResNet50 사용
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # 기본적으로 가중치 고정

# 모델 구성
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])

# 컴파일 및 학습
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(train_generator, epochs=10)

for layer in base_model.layers[-10:]:
    layer.trainable = True

optimizer_fine = SGD(learning_rate=1e-5, momentum=0.9)
model.compile(optimizer=optimizer_fine, loss="categorical_crossentropy", metrics=["accuracy"])

history_fine = model.fit(train_generator, epochs=30)

# 모델 저장
model_save_path = '/Users/soohyeon/Desktop/AI6_WorldAIProject/projectfile/resnet50_finetuned.h5'
model.save(model_save_path)
print(f"Fine-Tuning 완료 모델 저장됨: {model_save_path}")

# 학습 과정 시각화
def plot_training(history):
    acc = history.history["accuracy"]
    loss = history.history["loss"]
    epochs = range(len(acc))

    plt.figure(figsize=(10, 4))

    # 정확도 그래프
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, "b", label="Training Accuracy")
    plt.title("Training Accuracy")
    plt.legend()
    # 손실 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, "r", label="Training Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.show()

plot_training(history_fine)

# 테스트 이미지 예측 함수
def predict_images(model, test_dir, class_indices):
    class_labels = {v: k for k, v in class_indices.items()}

    for img_name in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img_name)
        if not img_name.lower().endswith(("png", "jpg", "jpeg")):
            continue  # 이미지 파일이 아니면 건너뛰기

        # 이미지 불러오기 및 전처리
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 예측 테스트 이미지 분류
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = class_labels[predicted_class]

        print(f"이미지: {img_name} -> 유형 : {predicted_label}")

# 테스트 이미지 예측 실행
predict_images(model, test_data_path, train_generator.class_indices)
