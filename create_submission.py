import os
import pandas as pd
from image_predictor import predict_image

# 기본 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../../data')
TEST_PATH = os.path.join(DATA_PATH, 'test')
SUB_CSV_PATH = os.path.join(DATA_PATH, 'sample_submission.csv')

submission = pd.read_csv(SUB_CSV_PATH)

# 예측 반복
for i, row in submission.iterrows():
    img_path = os.path.join(TEST_PATH, row['id'])

    if not os.path.exists(img_path):
        print(f"[경고] 이미지 없음: {img_path}")
        submission.at[i, 'label'] = 'unknown'
        continue

    pred_label, _ = predict_image(img_path, filter_type='clahe')
    submission.at[i, 'label'] = pred_label
    print(f"[{i+1}] {row['id']} → {pred_label}")

# 결과 저장
save_path = os.path.join(DATA_PATH, 'submission.csv')
submission.to_csv(save_path, index=False)
print(f"\n파일 저장 완료: {save_path}")