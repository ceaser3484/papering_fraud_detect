import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../../data/cnn_model_gradcam.h5')  # 모델 경로
TEST_IMAGE_PATH = os.path.join(BASE_DIR, '../../data/sample_image.jpeg')  # 테스트 이미지 경로

# 클래스 목록 (train_main.py 기준)
CLASSES = [
    '가구수정', '걸레받이수정', '곰팡이', '꼬임', '녹오염', '들뜸', '면불량', '몰딩수정', '반점', '석고수정',
    '오염', '오타공', '울음', '이음부불량', '창문,문틀수정', '터짐', '틈새과다', '피스', '훼손'
]

# 모델 로드
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError('모델 파일 존재하지 않음')
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# 대비 향상 함수 (히스토그램 평활화)
def enhance_contrast(img_rgb):
    img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    enhanced_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return enhanced_img

# 이미지 전처리
def preprocess_image(img_path, filter_type='clahe'):
    if not os.path.exists(img_path):
        raise FileNotFoundError('이미지 파일 존재하지 않음')

    img = Image.open(img_path).convert('RGB')
    img = np.array(img)

    # 대비 향상 적용
    img = enhance_contrast(img)

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if filter_type == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_gray)
        img_processed = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
    else:
        img_processed = img  # 필터 없이 향상된 RGB 사용

    img_resized = cv2.resize(img_processed, (224, 224))
    img_resized = img_resized.astype('float32') / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    return img_resized

# 예측 함수
def predict_image(img_path, filter_type='clahe'):
    img_input = preprocess_image(img_path, filter_type=filter_type)

    prediction = model.predict(img_input)[0]
    sorted_indices = np.argsort(prediction)[-2:]  # 상위 2개
    pred_index, second_pred_index = sorted_indices[-1], sorted_indices[-2]

    pred_prob = prediction[pred_index]
    second_pred_prob = prediction[second_pred_index]

    # 신뢰도 조정
    if pred_prob < 0.6 and second_pred_prob > 0.3:
        combined_score = (pred_prob * 0.7) + (second_pred_prob * 0.3)
        print(f"신뢰도 조정: {combined_score:.2f}")
        if combined_score > pred_prob:
            pred_prob = combined_score
            pred_index = second_pred_index

    print(f"예측 1: {CLASSES[pred_index]} ({pred_prob:.2f})")
    print(f"예측 2: {CLASSES[second_pred_index]} ({second_pred_prob:.2f})")

    if pred_prob < 0.5:
        print("예측 확률이 낮음, 추가 확인 필요")

    return CLASSES[pred_index], pred_prob

# 실행
if __name__ == '__main__':
    if os.path.exists(TEST_IMAGE_PATH):
        result, confidence = predict_image(TEST_IMAGE_PATH, filter_type='clahe')
        print(f'\n최종 예측 결과: {result} ({confidence:.2f})')
    else:
        print('이미지 파일이 존재하지 않음')