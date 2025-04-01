import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

train_dir = '/Users/soohyeon/Desktop/AI_project_2/open/train_augmented'
img_size = (224, 224)
batch_size = 32

# 데이터 증강 설정
datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.1,  # 가로 방향으로 이미지 이동
    height_shift_range=0.1, # 세로 방향으로 이미지 이동
    validation_split=0.2
)
# 훈련 데이터 제너레이터
train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)
# 검증 데이터 제너레이터
valid_gen = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers[:160]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=x)

model.compile(optimizer=Adam(learning_rate=0.00005), loss='categorical_crossentropy', metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  # patience 증가
checkpoint = ModelCheckpoint('best_model_size_aug.h5', monitor='val_loss', save_best_only=True)

callbacks = [early_stopping, checkpoint, reduce_lr]

history = model.fit(
    train_gen,
    epochs=50,  # 에폭 수 증가
    validation_data=valid_gen,
    callbacks=callbacks
)

model.save('final_model_size_aug.h5')

# import os
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from tensorflow.keras.layers import GlobalAveragePooling2D
# from tensorflow.keras.models import load_model
# # 추가 학습 데이터 경로
# fine_tune_data_path = "/Users/soohyeon/Desktop/AI_project/open/train_augmented"
# fine_tune_datagen = ImageDataGenerator(rescale=1.0 / 255)

# fine_tune_generator = fine_tune_datagen.flow_from_directory(
#     fine_tune_data_path,
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical'
# )

# fine_tune_model = load_model("AI6_project_model_classification_final.h5")

# base_model = fine_tune_model.layers[0]
# for layer in base_model.layers[:80]:
#     layer.trainable = False
# for layer in base_model.layers[80:]:
#     layer.trainable = True

# # 학습률 조정 (기존보다 낮은 값으로 조정)
# fine_tune_model.compile(
#     optimizer=Adam(learning_rate=0.00001),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"]
# )
# checkpoint_path = "AI6_project_finetuned_classification_model.h5"
# checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
# early_stopping = EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)

# # 추가 학습
# fine_tune_epochs = 40  # 추가 학습할 epoch 수
# history_fine_tune = fine_tune_model.fit(
#     fine_tune_generator,
#     epochs=fine_tune_epochs,
#     callbacks=[checkpoint, early_stopping, reduce_lr]
# )


# fine_tune_model.save("AI6_project_model_classification_finetuned.h5")
# print("추가 학습된 모델이 'AI6_project_model_classification_finetuned.h5'에 저장되었습니다.")

# def plot_training(history):
#     acc = history.history["accuracy"]
#     val_acc = history.history["val_accuracy"]
#     loss = history.history["loss"]
#     val_loss = history.history["val_loss"]
#     epochs_range = range(len(acc))

#     plt.figure(figsize=(10, 4))

#     # 정확도 그래프
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs_range, acc, "b", label="Training Accuracy")
#     plt.plot(epochs_range, val_acc, "g", label="Validation Accuracy")
#     plt.title("Training and Validation Accuracy")
#     plt.legend()

#     # 손실 그래프
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs_range, loss, "r", label="Training Loss")
#     plt.plot(epochs_range, val_loss, "orange", label="Validation Loss")
#     plt.title("Training and Validation Loss")
#     plt.legend()

#     plt.show()
# # 추가 학습 결과 시각화
# plot_training(history_fine_tune)
