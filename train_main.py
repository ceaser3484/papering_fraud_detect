import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback

train_dir = '/Users/soohyeon/Desktop/AI_project_2/open/train_augmented'
img_size = (224, 224)
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)


valid_gen = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)


base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Fine-tuning: ResNet50의 마지막 20개 레이어를 학습 가능하게 설정
for layer in base_model.layers[:160]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
dropout_layer = Dropout(0.5)
x = dropout_layer(x)
x = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=x)

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.00005), loss='categorical_crossentropy', metrics=['accuracy'])

class DynamicDropout(Callback):
    def __init__(self, layer, initial_rate=0.5, decay=0.005):
        super(DynamicDropout, self).__init__()
        self.layer = layer
        self.initial_rate = initial_rate
        self.decay = decay

    def on_epoch_begin(self, epoch, logs=None):
        new_rate = max(self.initial_rate - epoch * self.decay, 0.1)  # 최소 0.1로 제한
        self.layer.rate = new_rate
        print(f"Epoch {epoch+1}: Dropout rate updated to {new_rate:.4f}")

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
dynamic_dropout = DynamicDropout(dropout_layer, initial_rate=0.5, decay=0.005)

callbacks = [early_stopping, checkpoint, reduce_lr, dynamic_dropout]

# 모델 학습
history = model.fit(
    train_gen,
    epochs=20,
    validation_data=valid_gen,
    callbacks=callbacks
)

# 최종 모델 저장
model.save('final_model.h5')
