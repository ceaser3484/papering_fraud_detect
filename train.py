from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import pandas as pd
import numpy as np
from glob import glob
from sklearn.utils import class_weight
import pickle


def main():
    import re
    import os

    base_dir = "../../DATASET/mapping_img_data"
    if not os.path.isdir(os.path.join(base_dir, 'working_1')):
        print('sorry you should generate image. you should activate augumentation.py')
        exit()

    train = pd.DataFrame({'path':glob(os.path.join(base_dir, 'train','*','*'))})
    train['label'] = train['path'].apply(lambda x: x.split('/')[-2])

    train_aug = pd.DataFrame({'path':glob(os.path.join(base_dir, 'working_1','*','*'))})
    train_aug['label'] = train_aug['path'].apply(lambda x: x.split('/')[-1].split('.')[0]
                                                 ).apply(lambda x: re.sub(r"[0-9]","",x))

    train = pd.concat([train, train_aug], axis=0)

    test_data = pd.DataFrame({'path': glob(os.path.join(base_dir, 'test', '*'))})
    test_data['id'] = test_data['path'].apply(lambda x: 'TEST_' + x.split('/')[-1].split('.')[0])

    submission_data = pd.read_csv(os.path.join(base_dir, 'sample_submission.csv'))
    test_data = test_data.join(submission_data.set_index('id'), on='id')

    # edited = train[train['label'] == '걸레받이수정']
    # damaged = train[train['label'] == '훼손']
    # train.drop(train[(train['label'] == '훼손') | (train['label'] == '걸레받이수정')].index, inplace=True)
    #
    # sampled_emitted = edited.sample(n=600)
    # sampled_dammaged = damaged.sample(n=600)
    # del edited
    # del damaged
    # train = pd.concat([train, sampled_dammaged, sampled_emitted], axis=0)
    # del sampled_emitted
    # del sampled_dammaged

    kfold = StratifiedKFold(n_splits=3, random_state=3, shuffle=True)
    image_data_generator_train = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
    )
    image_data_generator_val = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )
    image_data_generator_test = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255
    )
    model = tf.keras.Sequential([
        tf.keras.applications.efficientnet.EfficientNetB7(include_top=False,pooling='avg'),
        # tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(19, activation='softmax')
    ])

    early_stop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(patience=5, verbose=1)

    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(train, train['label'])):
        print(f"{fold_idx} fold trainning")
        train_data = train.iloc[train_idx]
        val_data = train.iloc[val_idx]

        train_generator = image_data_generator_train.flow_from_dataframe(train_data, x_col='path',
                        class_mode='sparse',y_col='label',batch_size=4, target_size=(600,600))
        val_generator = image_data_generator_val.flow_from_dataframe(
            val_data, class_mode='sparse',x_col='path', y_col='label', batch_size=4, target_size=(600,600)
        )

        if fold_idx == 0:
            with open('fraud-class_origin.pkl', 'wb') as f:
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
                  validation_data=val_generator, epochs=10,
                  callbacks=[early_stop, reduce_lr])

        test_generate = image_data_generator_test.flow_from_dataframe(
            test_data, x_col='path', y_col='label', batch_size=4,  class_mode='sparse', target_size=(600,600)
        )

        model.evaluate(test_generate)

        model.evaluate()

        model.save("model_fold.keras")
if __name__ == '__main__':
    main()