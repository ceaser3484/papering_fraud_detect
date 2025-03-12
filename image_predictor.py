import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../../data/cnn_model_fixed.h5')
TEST_IMAGE_PATH = os.path.join(BASE_DIR, '../../data/sample_image.png')

# load model
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError('모델 파일 존재하지 않음')
    
    return tf.keras.models.load_model(MODEL_PATH, compile=True)

model = load_model()

# class 목록
CLASSES = [
    '가구수정', '걸레받이수정', '곰팡이', '꼬임', '녹오염', '들뜸', '면불량', '몰딩수정', '반점', '석고수정',
    '오염', '오타공', '울음', '이음부불량', '창문,문틀수정', '터짐', '틈새과다', '피스', '훼손'
]

# filter
def apply_filters(img_path, filter_type='clahe'):
    if not os.path.exists(img_path):
        raise FileNotFoundError('이미지 파일 존재하지 않음')
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    if filter_type == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
        processed_img = clahe.apply(img)
    else:
        processed_img = clahe.apply(img) # 기본 필터 사용 안할 시 원본사용
    
    return processed_img

# Top2
def predict_image(img_path, filter_type='clahe'):
    if not os.path.exists(img_path):
        raise FileNotFoundError('이미지 파일 존재하지 않음')
    
    filtered_img = apply_filters(img_path, filter_type=filter_type)
    filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)
    filtered_img = cv2.resize(filtered_img, (224, 224))
    filtered_img = filtered_img.astype('float32') / 225.0
    filtered_img = np.expand_dims(filtered_img, axis=0)
    
    prediction = model.predict(filtered_img)[0]
    sorted_indices = np.argsort(prediction)[-2:] # 상위 확률 2개
    pred_index, second_pred_index = sorted_indices[-1], sorted_indices[-2]
    
    pred_prob = prediction[pred_index]
    second_pred_prob = prediction[second_pred_index]
    
    print(f" {CLASSES[pred_index]} ({pred_prob:.2f})")
    print(f"{CLASSES[second_pred_index]} ({second_pred_prob:.2f})")
    
    if pred_prob < 0.5:
        print("예측 확률이 낮음, 두 번째 클래스도 참고 가능")
        
    return CLASSES[pred_index]

if __name__ == '__main__':
    if os.path.exists(TEST_IMAGE_PATH):
        result = predict_image(TEST_IMAGE_PATH, filter_type='clahe')
        print(f'\n 최종 예측 결과 : {result}')
    else:
        print(f'이미지 파일이 존재하지 않음')