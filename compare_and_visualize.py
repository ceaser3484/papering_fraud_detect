import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# macOS 한글 폰트 설정
font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
if os.path.exists(font_path):
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
else:
    print("한글 폰트가 감지되지 않음")

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'data')
SUBMISSION_PATH = os.path.join(DATA_DIR, 'submission.csv')
SAMPLE_PATH = os.path.join(DATA_DIR, 'sample_submission.csv')

# CSV 불러오기
submission_df = pd.read_csv(SUBMISSION_PATH)
sample_df = pd.read_csv(SAMPLE_PATH)

# 정렬 및 비교
submission_df = submission_df.sort_values("id").reset_index(drop=True)
sample_df = sample_df.sort_values("id").reset_index(drop=True)

comparison = submission_df.copy()
comparison['ground_truth'] = sample_df['label']
comparison['is_correct'] = comparison['label'] == comparison['ground_truth']

# 요약 통계
total = len(comparison)
correct = comparison['is_correct'].sum()
accuracy = correct / total * 100

print(f" 총 {total}개 중 {correct}개 정답 → 정확도: {accuracy:.2f}%")

# 예측 정확도 시각화
plt.figure(figsize=(6, 4))
plt.bar(['정답', '오답'], [correct, total - correct], color=['green', 'red'])
plt.title('예측 결과 비교')
plt.ylabel('개수')
plt.tight_layout()
plt.show()

# 신뢰도 기반 추가 분석
if 'confidence' in submission_df.columns:
    print("\n신뢰도 분석:")
    low_conf_df = comparison[submission_df['confidence'] < 0.5]
    print(f"신뢰도 0.5 미만 예측: {len(low_conf_df)}개")

    if not low_conf_df.empty:
        print("\n낮은 신뢰도 예측 샘플:")
        print(low_conf_df[['id', 'label', 'ground_truth', 'confidence']].head())

        # 히스토그램
        plt.figure(figsize=(6, 4))
        plt.hist(low_conf_df['confidence'], bins=10, color='orange', edgecolor='black')
        plt.title("신뢰도 낮은 예측 분포")
        plt.xlabel("신뢰도")
        plt.ylabel("이미지 수")
        plt.tight_layout()
        plt.show()