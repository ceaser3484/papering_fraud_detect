import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import tensorflow as tf
import os
import numpy as np
import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def start_app():
    image_url = 'https://img1.daumcdn.net/thumb/R1280x0/' \
                '?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%' \
                '2Fdn%2Feuxjuu%2FbtqCbS7n1MC%2FEopTZApFrckDvSWlrQqfH1%2Fimg.jpg'

    # print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    print('this is the first test app')
    st.title('this is testing app')
    st.write("Hello World")
    st.write('어서오세요 새로운 세상에')
    st.image(image_url)

    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    tf_image = tf.keras.utils.img_to_array(image)

    tf_image = tf.image.resize(tf_image, [528,528])
    tf_image = tf_image / 255.0
    tf_image = np.expand_dims(tf_image, axis=0)

    model = tf.keras.models.load_model('b6_best_model_fold.keras')
    result = model.predict(tf_image)
    label = np.argmax(result, axis=1)

    with open('fraud-class_origin.pkl', 'rb') as f:
        pre_classes = pickle.load(f)
    classes = {value: key for key, value in pre_classes.items()}
    # print(type(classes))
    # print(label[0])
    st.write(f"현재 도배 상태는 {classes[label[0]]}으로 도배를 다시 요청하세요")

if __name__ == '__main__':
    start_app()