import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../../data')
MODEL_PATH = os.path.join(DATA_PATH, 'cnn_model_weighted.h5')
TEST_DIR = os.path.join(DATA_PATH, 'dataset', 'test')
SUBMISSION_PATH = os.path.join(DATA_PATH, 'submission.csv')

# 클래스 목록
CLASSES = [
    '가구수정', '걸레받이수정', '곰팡이', '꼬임', '녹오염', '들뜸', '면불량', '몰딩수정', '반점', '석고수정',
    '오염', '오타공', '울음', '이음부불량', '창문,문틀수정', '터짐', '틈새과다', '피스', '훼손'
]

# 모델 불러오기
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# 이미지 전처리 함수
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)

# 예측 수행
results = []
file_names = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".png")])

for file_name in file_names:
    img_path = os.path.join(TEST_DIR, file_name)

    try:
        img_array = preprocess_image(img_path)
        predictions = model.predict(img_array)[0]
        pred_index = np.argmax(predictions)
        pred_class = CLASSES[pred_index]
        confidence = float(predictions[pred_index])  # 소수점 변환
        results.append((file_name, pred_class, confidence))
    except Exception as e:
        print(f"[경고] 이미지 처리 실패: {file_name} -> {e}")
        results.append((file_name, "예측불가", 0.0))

# CSV 저장
submission_df = pd.DataFrame(results, columns=["id", "label", "confidence"])
submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"[완료] 파일 저장: {SUBMISSION_PATH}")