import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import random
from torchvision import transforms  # type: ignore
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.optim as optim

data_dir = "/Users/soohyeon/Desktop/AI_project/open/train"

def augment_to_target_count(input_dir, target_count=2000):
    """
    input_dir : 원본 이미지 루트 폴더 경로
    target_count : 각 폴더에 생성할 목표 이미지 수
    """
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
        A.Resize(224, 224),
        ToTensorV2()
    ])

    sub_folders = [f.path for f in os.scandir(input_dir) if f.is_dir()]

    for sub_folder in sub_folders:
        folder_name = os.path.basename(sub_folder)
        output_folder = os.path.join(os.path.dirname(input_dir), "train_augmented", folder_name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    for sub_folder in sub_folders:
        folder_name = os.path.basename(sub_folder)
        input_folder = sub_folder
        output_folder = os.path.join(os.path.dirname(input_dir), "train_augmented", folder_name)

        # 이미지 파일 목록 가져오기
        image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        original_count = len(image_files)

        if original_count >= target_count:
            print(f"Skipping {folder_name}: Already has {original_count} images.")
            continue

        augmentations_needed = target_count - original_count

        # 랜덤 이미지 선택 및 증강
        for i in range(augmentations_needed):
            random_image_file = random.choice(image_files)
            image_path = os.path.join(input_folder, random_image_file)
            image = Image.open(image_path).convert('RGB')

            transformed = train_transforms(image=np.array(image))['image']
            augmented_image = Image.fromarray(transformed.numpy().transpose(1, 2, 0).astype('uint8'))

            new_image_name = f"{os.path.splitext(random_image_file)[0]}_augmented_{i}.jpg"
            new_image_path = os.path.join(output_folder, new_image_name)
            augmented_image.save(new_image_path)

            print(f"Augmented image saved: {new_image_path}")

input_root_directory = "/Users/soohyeon/Desktop/AI_project/open/train"  # 원본 이미지 루트 폴더 경로
augment_to_target_count(input_root_directory, target_count=1000)  # 각 폴더

data_dir_augmented = "/Users/soohyeon/Desktop/AI_project/open/train_augmented"

for folder in os.listdir(data_dir_augmented):
    path = os.path.join(data_dir_augmented, folder)
    if os.path.isdir(path):
        print(f"{folder}: {len(os.listdir(path))}장")
