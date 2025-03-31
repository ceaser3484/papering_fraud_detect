import os
import random
import numpy as np
from PIL import Image
import albumentations as A

# 원본 이미지 폴더와 증강 이미지 저장 폴더 경로
data_dir = "/Users/soohyeon/Desktop/AI_project_2/open/train"
output_dir_parent = "/Users/soohyeon/Desktop/AI_project_2/open/train_augmented_1000"

# 각 폴더당 1,000장 목표
TARGET_COUNT = 1000

def augment_images_to_target(input_dir, output_dir_parent, target_count=TARGET_COUNT):
    """
    input_dir : 원본 이미지 루트 폴더 경로
    output_dir_parent : 증강된 이미지를 저장할 부모 폴더 경로
    target_count : 각 폴더당 최종 이미지 개수
    """
    # 증강을 위한 변환 리스트
    transform = A.Compose([
        A.ShiftScaleRotate(p=0.5, shift_limit=0.1, scale_limit=0.1, rotate_limit=15),  # 이동, 스케일 조절, 회전
        A.GridDistortion(p=0.3),
        A.HueSaturationValue(p=0.5, hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10),  # 색조 변경
        A.CLAHE(p=0.3, clip_limit=2.0, tile_grid_size=(8, 8)),  # 대비 향상
        A.CoarseDropout(p=0.3, max_holes=3, max_height=32, max_width=32),  # 특정 영역 마스킹
        A.Equalize(p=0.3)  # 명암 균등화
    ])

    # 각 클래스 폴더 확인
    sub_folders = [f.path for f in os.scandir(input_dir) if f.is_dir()]

    for sub_folder in sub_folders:
        folder_name = os.path.basename(sub_folder)
        output_folder = os.path.join(output_dir_parent, folder_name)
        os.makedirs(output_folder, exist_ok=True)

        # 원본 이미지 파일 리스트
        image_files = [f for f in os.listdir(sub_folder) if f.endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('.')]
        original_count = len(image_files)

        # 목표 개수 이상이면 증강하지 않음
        if original_count >= target_count:
            print(f"✅ {folder_name}: {original_count}장 (이미 목표 이상)")
            continue

        # 부족한 개수 계산
        needed_images = target_count - original_count
        print(f"🔹 {folder_name}: {original_count}장 → {target_count}장 (필요한 증강 수: {needed_images})")

        # 원본 이미지가 없으면 증강할 수 없음
        if not image_files:
            print(f"{folder_name} 폴더에 이미지가 없습니다. 증강 불가능.")
            continue

        # 랜덤하게 원본 이미지를 선택하여 부족한 개수만큼 증강
        augmented_count = 0
        while augmented_count < needed_images:
            image_file = random.choice(image_files)  # 랜덤 이미지 선택
            image_path = os.path.join(sub_folder, image_file)

            try:
                image = Image.open(image_path).convert('RGB')

                transformed = transform(image=np.array(image))['image']
                augmented_image = Image.fromarray(transformed.astype('uint8'))

                new_image_name = f"{os.path.splitext(image_file)[0]}_aug_{augmented_count}.jpg"
                new_image_path = os.path.join(output_folder, new_image_name)
                augmented_image.save(new_image_path)

                augmented_count += 1

            except Exception as e:
                print(f"{image_file} 증강 중 오류 발생: {e}")

        print(f"{folder_name} 증강 완료: 총 {target_count}장")

    print("모든 폴더의 증강이 완료되었습니다.")

# 실행
augment_images_to_target(data_dir, output_dir_parent)

# 각 폴더에 저장된 이미지 개수 확인
for folder in os.listdir(output_dir_parent):
    path = os.path.join(output_dir_parent, folder)
    if os.path.isdir(path):
        print(f"📂 {folder}: {len(os.listdir(path))}장")
