
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.src.layers import BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.layers import *
from sklearn.utils import class_weight
from tensorflow.python.keras.layers import Flatten


def preprocessing(img):
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    cv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    cv_img = list(cv2.split(cv_img))
    cv_img[0] = cv2.equalizeHist(cv_img[0])
    cv_img = cv2.merge(cv_img)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_YCrCb2RGB)

    # scaling and shift data type
    img_float = cv_img.astype(np.float32) / 255.0
    return img_float

def main():
    batch_size = 2
    images_path = '/home/ceaser/DATASET/mapping_img_data/train/'
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocessing,
         horizontal_flip=True, fill_mode='nearest',
        vertical_flip=True, zoom_range=[0.3, 0.5],
        validation_split=0.2
    )
    train_image_data = image_generator.flow_from_directory(
        images_path, subset='training', target_size=(224,224), class_mode='sparse',
        batch_size=batch_size
    )

    val_image_data = image_generator.flow_from_directory(
        images_path, subset='validation', target_size=(224, 224), class_mode='sparse',
        batch_size=batch_size
    )


    train_class_weight = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(train_image_data.classes),
        y=train_image_data.classes
    )
    train_class_weight = dict(enumerate(train_class_weight))

    resnet = tf.keras.applications.ResNet50V2(
        include_top=False, input_shape=(224,224,3)
    )
    model = tf.keras.models.Sequential([
        resnet,
        Dense(512, activation='relu'),
        BatchNormalization(),
        GlobalAveragePooling2D(),
        Dense(19, activation='softmax')
    ])

    print(model.summary())

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.00001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_image_data,batch_size=8, validation_data=val_image_data,
              class_weight=train_class_weight, epochs=100)

    # model.fit_generator(
    #     train_image_data,
    #     steps_per_epoch=train_image_data.sample // batch_size,
    #     validation_data=val_image_data,
    #     validation_steps=val_image_data.sample // batch_size,
    #     epochs=100
    # )
if __name__ == '__main__':
    main()
