from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import pandas as pd
import numpy as np
from glob import glob
from sklearn.utils import class_weight
import pickle


def main():
    import re
    from os.path import isdir

    if not isdir("../../DATASET/mapping_img_data/working/"):
        print('sorry you should generate image. you should activate augumentation.py')
        exit()

    train = pd.DataFrame({'path':glob("../../DATASET/mapping_img_data/train/*/*")})
    train['label'] = train['path'].apply(lambda x: x.split('/')[-2])

    train_aug = pd.DataFrame({'path':glob('../../DATASET/mapping_img_data/working/*/*')})
    train_aug['label'] = train_aug['path'].apply(lambda x: x.split('/')[-1].split('.')[0]
                                                 ).apply(lambda x: re.sub(r"[0-9]","",x))

    train = pd.concat([train, train_aug], axis=0)

    edited = train[train['label'] == '걸레받이수정']
    damaged = train[train['label'] == '훼손']
    train.drop(train[(train['label'] == '훼손') | (train['label'] == '걸레받이수정')].index, inplace=True)

    sampled_emitted = edited.sample(n=600)
    sampled_dammaged = damaged.sample(n=600)
    del edited
    del damaged
    train = pd.concat([train, sampled_dammaged, sampled_emitted], axis=0)
    del sampled_emitted
    del sampled_dammaged

    kfold = StratifiedKFold(n_splits=5, random_state=3, shuffle=True)
    image_data_generator_train = tf.keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True, vertical_flip=True, rescale=1./255
    )
    image_data_generator_val = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )

    model = tf.keras.Sequential([
        tf.keras.applications.efficientnet.EfficientNetB4(include_top=False,pooling='avg'),
        tf.keras.layers.Dense(19, activation='softmax')
    ])

    early_stop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(patience=5, verbose=1)

    model.compile(optimizer='adamw',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    for train_idx, val_idx in kfold.split(train, train['label']):
        train_data = train.iloc[train_idx]
        val_data = train.iloc[val_idx]

        train_generator = image_data_generator_train.flow_from_dataframe(train_data, x_col='path',
                        class_mode='sparse',y_col='label',batch_size=32, target_size=(300,300))
        val_generator = image_data_generator_val.flow_from_dataframe(
            val_data, class_mode='sparse',x_col='path', y_col='label', batch_size=32, target_size=(300,300)
        )

        with open('fraud-class.pkl', 'wb') as f:
            pickle.dump(train_generator.class_indices, f)


        train_class_weight = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(train_generator.classes),
            y=train_generator.classes
        )
        train_class_weight = dict(enumerate(train_class_weight))

        # model.compile(optimizer='rmsprop',
        #               loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_generator, class_weight=train_class_weight,
                  validation_data=val_generator, epochs=50,
                  callbacks=[early_stop, reduce_lr])

        model.save("model_fold.keras")
if __name__ == '__main__':
    main()