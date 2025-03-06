import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../../data')
train_data_path = os.path.join(DATA_PATH, 'dataset', 'train')

# 증강
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.3,
    zoom_range = 0.3,
    horizontal_flip = True,
    brightness_range = [0.7, 1.3]
)

# load data
train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size = (224, 224),
    batch_size = 16,
    class_mode = 'categorical'
)

num_classes = len(train_generator.class_indices)

# 가중치
class_counts = np.bincount(train_generator.classes)
class_weights = compute_class_weight('balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
class_weight_dict = dict(enumerate(class_weights))

# ResNet50
base_model = ResNet50(weights = 'imagenet', include_top = False, input_shape = (224, 224, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation = 'relu'),
    Dropout(0.4),
    Dense(num_classes, activation = 'softmax')
])

# 학습률 조정
lr_reduction = ReduceLROnPlateau(monitor = 'loss', patience = 3, factor = 0.5, min_lr = 1e-6)

# checkpoint
checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor = 'val_accuracy',
    save_best_only = True,
    mode = 'max'
)

# compile
model.compile(
    optimizer = Adam(learning_rate = 0.0001),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

# 학습
history = model.fit(
    train_generator,
    epochs = 50,
    class_weight = class_weight_dict,
    callbacks = [lr_reduction, checkpoint]
)

model_save_path = os.path.join(DATA_PATH, 'cnn_model_fixed.h5')
model.save(model_save_path)
print('모델 학습 및 저장 완료')