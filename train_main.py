import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

train_dir = '/Users/soohyeon/Desktop/AI6_WorldAIProject/projectfile/train'  # train 데이터가 있는 폴더 경로
test_dir = '/Users/soohyeon/Desktop/AI6_WorldAIProject/projectfile/test'  # test 데이터가 있는 폴더 경로

# 데이터 증강 및 정규화
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 2. 모델 설정 ResNet50 사용
model = models.resnet50(pretrained=True)
# 마지막 레이어 수정 (분류 클래스 수에 맞게)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))

# 모델을 GPU로 이동
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()  # Sparse Categorical Cross Entropy는 기본적으로 CrossEntropyLoss로 처리됨
optimizer = optim.Adam(model.parameters(), lr=0.001)
#모델훈련
num_epochs = 20  # Epochs 20으로 설정
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # 모델 예측
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

model.eval()
all_preds = []
all_labels = []
all_probs = []

test_images = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
for img_name in tqdm(test_images, desc="Testing"):
    img_path = os.path.join(test_dir, img_name)
    img = Image.open(img_path).convert('RGB')

    # 변환 적용
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        
        all_labels.extend([0])

        # ROC-AUC를 위해 확률값 저장
        probs = torch.softmax(outputs, dim=1)
        all_probs.extend(probs.cpu().numpy()[:, 1])  # "하자 있음" 클래스에 대한 확률

accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

precision = precision_score(all_labels, all_preds, average='binary') 
recall = recall_score(all_labels, all_preds, average='binary')
f1 = f1_score(all_labels, all_preds, average='binary')
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

roc_auc = roc_auc_score(all_labels, all_probs)
print(f"ROC-AUC: {roc_auc:.2f}")

fpr, tpr, _ = roc_curve(all_labels, all_probs)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
