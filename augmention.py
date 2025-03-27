import os
import random
import numpy as np
from PIL import Image
import albumentations as A

data_dir = "/Users/soohyeon/Desktop/AI_project/open/train"
output_dir_parent = "/Users/soohyeon/Desktop/AI_project/open/train_augmented"

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# def equalize_image(img):
#     img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     img_equalized = cv2.equalizeHist(img_gray)
#     img_equalized = cv2.cvtColor(img_equalized, cv2.COLOR_GRAY2RGB)
#     return img_equalized

# def apply_clahe(img):
#     img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 이미지를 그레이스케일로 변환
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # CLAHE 객체 생성
#     img_clahe = clahe.apply(img_gray)  # CLAHE 적용
#     img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)  # 다시 RGB로 변환
#     return img_clahe

def augment_to_target_count(input_dir, output_dir_parent, target_count=1000):
    """
    input_dir : 원본 이미지 루트 폴더 경로
    output_dir_parent : 증강된 이미지를 저장할 폴더 경로
    target_count : 각 폴더에 생성할 목표 이미지 수
    """
    transform = A.Compose([
        A.Blur(p=0.3, blur_limit=(2, 5)),
        A.ElasticTransform(p=0.5, alpha=1.0, sigma=50.0, alpha_affine=50.0),
        A.GaussNoise(p=0.5, var_limit=(10.0, 50.0), per_channel=True),
        A.RandomFog(p=0.5, fog_coef_lower=0.1, fog_coef_upper=0.2, alpha_coef=0.08),
        A.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
        A.RandomGamma(p=0.5, gamma_limit=(90, 110)),
        A.HorizontalFlip(p=0.5)
    ])

    resize_transform = A.Compose([
        A.Resize(224, 224)
    ])

    sub_folders = [f.path for f in os.scandir(input_dir) if f.is_dir()]

    for sub_folder in sub_folders:
        folder_name = os.path.basename(sub_folder)
        output_folder = os.path.join(output_dir_parent, folder_name)
        os.makedirs(output_folder, exist_ok=True)

    for sub_folder in sub_folders:
        folder_name = os.path.basename(sub_folder)
        input_folder = sub_folder
        output_folder = os.path.join(output_dir_parent, folder_name)

        image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('.')]
        original_count = len(image_files)

        for image_file in image_files:
            try:
                image_path = os.path.join(input_folder, image_file)
                image = Image.open(image_path).convert('RGB')
                
                resized_image = resize_transform(image=np.array(image))['image']
                resized_pil_image = Image.fromarray(resized_image.astype('uint8')) 
                new_image_path = os.path.join(output_folder, image_file)
                resized_pil_image.save(new_image_path)
                print(f"Resized original image saved: {new_image_path}")
            except Exception as e:
                print(f"Error resizing {image_file}: {e}")

        if original_count >= target_count:
            print(f"Skipping {folder_name}: Already has {original_count} images.")
            continue

        augmentations_needed = target_count - original_count

        if image_files:
            for i in range(augmentations_needed):
                try:
                    random_image_file = random.choice(image_files)
                    image_path = os.path.join(input_folder, random_image_file)
                    image = Image.open(image_path).convert('RGB')

                    transformed = transform(image=np.array(image))['image']
                    augmented_image = Image.fromarray(transformed.astype('uint8'))

                    new_image_name = f"{os.path.splitext(random_image_file)[0]}_augmented_{i}.jpg"
                    new_image_path = os.path.join(output_folder, new_image_name)
                    augmented_image.save(new_image_path)

                    print(f"Augmented image saved: {new_image_path}")
                except Exception as e:
                    print(f"Error augmenting image {random_image_file}: {e}")
        else:
            print(f"Error: {input_folder} 폴더에 이미지 파일이 없습니다.")

augment_to_target_count(data_dir, output_dir_parent, target_count=2000)

for folder in os.listdir(output_dir_parent):
    path = os.path.join(output_dir_parent, folder)
    if os.path.isdir(path):
        print(f"{folder}: {len(os.listdir(path))}장")
