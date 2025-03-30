import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# macOS 한글 폰트 설정
font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
if os.path.exists(font_path):
    plt.rcParams["font.family"] = fm.FontProperties(fname=font_path).get_name()
plt.rcParams["axes.unicode_minus"] = False  # 음수 기호 깨짐 방지

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../../data")
submission_path = os.path.join(DATA_PATH, "submission.csv")
sample_path = os.path.join(DATA_PATH, "sample_submission.csv")

# CSV 불러오기
submission_df = pd.read_csv(submission_path)
sample_df = pd.read_csv(sample_path)

# ID 정렬 및 정합성
submission_df = submission_df.sort_values("id").reset_index(drop=True)
sample_df = sample_df.sort_values("id").reset_index(drop=True)

if not all(submission_df["id"] == sample_df["id"]):
    print("[경고] submission과 sample의 ID가 정렬되어 있지않음")

# 비교용 데이터프레임
comparison = submission_df.copy()
comparison["ground_truth"] = sample_df["label"]
comparison["is_correct"] = comparison["label"] == comparison["ground_truth"]

# 총 개수 및 정확도 출력
total = len(comparison)
correct = comparison["is_correct"].sum()
accuracy = correct / total * 100
print(f"[요약] 총 {total}개 중 정답 {correct}개 → 정확도: {accuracy:.2f}%")

# Confusion Matrix 
y_true = comparison["ground_truth"]
y_pred = comparison["label"]
labels = sorted(list(set(y_true) | set(y_pred)))

cm = confusion_matrix(y_true, y_pred, labels=labels)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="Blues")
plt.xlabel("예측 값")
plt.ylabel("실제 정답")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(DATA_PATH, "confusion_matrix.png"))
plt.show()

# Classification Report 
report = classification_report(y_true, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# 평균 및 정확도 제외
filtered_df = report_df.loc[~report_df.index.str.contains("avg|accuracy")]

plt.figure(figsize=(12, 5))
sns.barplot(x=filtered_df.index, y=filtered_df["precision"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("정확도 (Precision)")
plt.title("클래스별 Precision")
plt.tight_layout()
plt.savefig(os.path.join(DATA_PATH, "class_precision.png"))
plt.show()

# F1 Score
plt.figure(figsize=(12, 5))
sns.barplot(x=filtered_df.index, y=filtered_df["f1-score"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("F1 Score")
plt.title("클래스별 F1 Score")
plt.tight_layout()
plt.savefig(os.path.join(DATA_PATH, "class_f1score.png"))
plt.show()