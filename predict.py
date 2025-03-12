import tensorflow as tf
import os
from glob import glob
import pandas as pd
import numpy as np
import pickle

def main():
    base_dir = "../../DATASET/mapping_img_data"
    test_data = pd.DataFrame({'path': glob(os.path.join(base_dir, 'test', '*'))})
    test_data['id'] = test_data['path'].apply(lambda x:  'TEST_'+ x.split('/')[-1].split('.')[0])

    submission_data = pd.read_csv(os.path.join(base_dir, 'sample_submission.csv'))
    test_data = test_data.join(submission_data.set_index('id'), on='id')
    test_data.sort_values(by=['id']) # 여기서 잘 안됨... 왜 sorting이 안될까?

    image_data_generator_predict = tf.keras.preprocessing.image.ImageDataGenerator(
       rescale=1. / 255
    )

    test = image_data_generator_predict.flow_from_dataframe(test_data, x_col='path', y_col='label', class_mode='sparse')
    model = tf.keras.models.load_model('model_fold.keras')
    result = model.predict(test)

    with open('fraud-class.pkl', 'rb') as f:
        classes = pickle.load(f)
    classes = {value:key for key, value in classes.items()}
    # print(classes)
    for fraud in np.argmax(result, axis=1):
        print(classes[fraud])




if __name__ == '__main__':
    main()
