import os
import random
import numpy as np
from PIL import Image
import albumentations as A

data_dir = "/Users/soohyeon/Desktop/AI_project_2/open/train"
output_dir_parent = "/Users/soohyeon/Desktop/AI_project_2/open/train_augmented_new_second"

def augment_to_target_count(input_dir, output_dir_parent, augment_count_per_image=4):
    """
    input_dir : 원본 이미지 루트 폴더 경로
    output_dir_parent : 증강된 이미지를 저장할 부모 폴더 경로
    augment_count_per_image : 각 이미지당 생성할 증강 이미지 수
    """
    # 증강을 위한 변환 리스트
    transform = A.Compose([
        A.ShiftScaleRotate(p=0.5, shift_limit=0.1, scale_limit=0.1, rotate_limit=15),  # 이동, 스케일 조절, 회전
        A.GridDistortion(p=0.3),
        A.HueSaturationValue(p=0.5, hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10),  # 색조 변경
        A.CLAHE(p=0.3, clip_limit=2.0, tile_grid_size=(8, 8)),  # CLAHE 대비 향상
        A.CoarseDropout(p=0.3, max_holes=3, max_height=32, max_width=32),  # 특정 영역 마스킹
        A.Equalize(p=0.3)  # 명암 균등화
    ])
        

    sub_folders = [f.path for f in os.scandir(input_dir) if f.is_dir()]

    # 각 폴더에 대해 증강 작업을 실행
    for sub_folder in sub_folders:
        folder_name = os.path.basename(sub_folder)
        output_folder = os.path.join(output_dir_parent, folder_name)
        os.makedirs(output_folder, exist_ok=True)

        image_files = [f for f in os.listdir(sub_folder) if f.endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('.')]
        original_count = len(image_files)

        # 증강된 이미지만 생성하고, 원본 이미지는 저장하지 않음
        if image_files:
            for image_file in image_files:
                try:
                    image_path = os.path.join(sub_folder, image_file)
                    image = Image.open(image_path).convert('RGB')

                    # 이미지 증강
                    for i in range(augment_count_per_image):
                        transformed = transform(image=np.array(image))['image']
                        augmented_image = Image.fromarray(transformed.astype('uint8'))

                        new_image_name = f"{os.path.splitext(image_file)[0]}_augmented_{i}.jpg"
                        new_image_path = os.path.join(output_folder, new_image_name)
                        augmented_image.save(new_image_path)

                    print(f"Augmented images for {image_file} saved.")
                except Exception as e:
                    print(f"Error augmenting image {image_file}: {e}")
        else:
            print(f"Error: {sub_folder} 폴더에 이미지 파일이 없습니다.")

    print("All augmentations are completed.")

augment_to_target_count(data_dir, output_dir_parent, augment_count_per_image=4)

# 각 폴더에 저장된 이미지 갯수 확인
for folder in os.listdir(output_dir_parent):
    path = os.path.join(output_dir_parent, folder)
    if os.path.isdir(path):
        print(f"{folder}: {len(os.listdir(path))}장")
