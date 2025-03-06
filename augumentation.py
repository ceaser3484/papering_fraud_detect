from PIL import Image
import cv2
import albumentations as A
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def manipulate_image(images_df, image_dir_name, num_times, augmented, *image_category, ):
    from tqdm import tqdm
    base_dir = '../../DATASET/mapping_img_data'
    os.makedirs(os.path.join(base_dir, 'working',image_dir_name), exist_ok=True)


    for category in image_category:
        for path in tqdm(images_df[images_df['label'] == category]['path']):
            for i in range(num_times):
                name = path.split('/')[-1]
                image = cv2.imread(path)
                image_augmented = augmented(image=image)['image']
                cv2.imwrite(os.path.join(base_dir, 'working',image_dir_name,
                                         str(i) + category + name), image_augmented)


def data_augument():
    from tqdm import tqdm

    train = pd.DataFrame({'path': glob.glob("../../DATASET/mapping_img_data/train/*/*")})
    test = pd.read_csv('../../DATASET/mapping_img_data/test.csv')
    train['label'] = train['path'].apply(lambda x: x.split('/')[-2])

    aug = A.Compose([
        A.VerticalFlip(),
        A.Rotate(p=0.7),
        A.HorizontalFlip(),
        A.RandomBrightnessContrast(brightness_limit=0.2),
        A.Resize(300,300)
    ])
    manipulate_image(train, 'train_302', 2, aug, '석고수정','들뜸','피스')
    print()
    manipulate_image(train, 'train_303', 3, aug,
                     '창틀,문틀수정','울음','이음부불량','녹오염','가구수정')
    print()
    manipulate_image(train, 'train_304', 6, aug,
                     '틈새과다','반점')

def show_num_classes():
    import seaborn as sns
    import matplotlib.font_manager
    import matplotlib as mpl

    train = pd.DataFrame({'path': glob.glob("../../DATASET/mapping_img_data/train/*/*")})
    train['label'] = train['path'].apply(lambda x: x.split('/')[-2])
    plt.rc('font',family='NanumGothic')
    plt.rcParams["font.family"] = 'NanumGothic'
    mpl.rcParams['axes.unicode_minus'] = False  # 마이너스 폰트 깨짐 방지

    # 예시 그래프
    plt.figure(figsize=(8, 6))
    plt.plot([1, 2, 3, 4], [10, 20, 25, 30])
    plt.title('예제 그래프')
    plt.xlabel('X 축')
    plt.ylabel('Y 축')
    plt.show()

    # label_count.plot(kind='bar', figsize=(10,10))
    # plt.show()
if __name__ == '__main__':
    data_augument()