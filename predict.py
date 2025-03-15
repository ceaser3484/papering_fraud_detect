import tensorflow as tf
import os
from glob import glob
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt


def main():
    base_dir = "../../DATASET/mapping_img_data"
    test_data = pd.DataFrame({'path': glob(os.path.join(base_dir, 'test', '*'))})
    test_data['id'] = test_data['path'].apply(lambda x:  'TEST_'+ x.split('/')[-1].split('.')[0])

    submission_data = pd.read_csv(os.path.join(base_dir, 'sample_submission.csv'))
    test_data = test_data.join(submission_data.set_index('id'), on='id')
    test_data = test_data.sample(n=4)
    print(test_data)

    image_data_generator_predict = tf.keras.preprocessing.image.ImageDataGenerator(
       rescale=1. / 255
    )

    test = image_data_generator_predict.flow_from_dataframe(test_data, x_col='path', y_col='label', class_mode='sparse')
    model = tf.keras.models.load_model('blur_model_fold_origin.keras')
    result = model.predict(test)

    with open('fraud-class_origin.pkl', 'rb') as f:
        classes = pickle.load(f)
    classes = {value:key for key, value in classes.items()}
    predicted_label = []
    for fraud in np.argmax(result, axis=1):
        predicted_label.append(classes[fraud])

    test_data = test_data.reset_index(drop=True)
    test_data['predicted'] = pd.Series(predicted_label, index=test_data.index)
    print(test_data)

    plt.rcParams['font.family'] = 'NanumGothic'  # 원하는 폰트명으로 변경
    plt.rcParams['axes.unicode_minus'] = False

    for idx, row in test_data.iterrows():
        img = tf.keras.utils.img_to_array(Image.open(row['path']))
        plt.imshow(img.astype(np.uint8))
        plt.title(f"label: {row['label']}\npredicted:{row['predicted']}")
        plt.show()

if __name__ == '__main__':
    main()
