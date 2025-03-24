import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# macOS에서 한글 폰트 설정
font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
if os.path.exists(font_path):
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
else:
    print("한글 폰트가 감지되지 않음")

SUBMISSION_PATH = "/Users/leemiinjeong/Desktop/Wall드AI/data/submission.csv"
SAMPLE_PATH = "/Users/leemiinjeong/Desktop/Wall드AI/data/sample_submission.csv"

# CSV 파일 불러오기
submission_df = pd.read_csv(SUBMISSION_PATH)
sample_df = pd.read_csv(SAMPLE_PATH)

# ID 정렬
submission_df = submission_df.sort_values("id").reset_index(drop=True)
sample_df = sample_df.sort_values("id").reset_index(drop=True)

# 비교
comparison = submission_df.copy()
comparison['ground_truth'] = sample_df['label']
comparison['is_correct'] = comparison['label'] == comparison['ground_truth']

# 요약 통계 출력
total = len(comparison)
correct = comparison['is_correct'].sum()
accuracy = correct / total * 100

print(f"총 {total}개 중 {correct}개 정답 정확도: {accuracy:.2f}%")

# 시각화
plt.figure(figsize=(6, 4))
plt.bar(['정답', '오답'], [correct, total - correct], color=['green', 'red'])
plt.title('예측 결과 비교')
plt.ylabel('개수')
plt.tight_layout()
plt.show()