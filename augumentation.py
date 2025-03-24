import cv2
import albumentations as A
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.sparse.csgraph import maximum_bipartite_matching


def manipulate_image(images_df, image_dir_name, num_times, augmented, category):
    from tqdm import tqdm
    base_dir = '../../DATASET/mapping_img_data'
    os.makedirs(os.path.join(base_dir, 'working',image_dir_name), exist_ok=True)

    for path in tqdm(images_df[images_df['label'] == category]['path']):
        for i in range(num_times):
            name = path.split('/')[-1]
            image = cv2.imread(path)
            image_augmented = augmented(image=image)['image']
            cv2.imwrite(os.path.join(base_dir, 'working',image_dir_name,
                                     str(i) + category + name), image_augmented)


def data_augument():


    train = pd.DataFrame({'path': glob.glob("../../DATASET/mapping_img_data/train/*/*")})

    train['label'] = train['path'].apply(lambda x: x.split('/')[-2])

    # aug = A.Compose([
    #     # A.CLAHE(p=0.3),
    #     A.AdvancedBlur(p=0.6),
    #     A.VerticalFlip(),
    #     A.Rotate(p=0.7),
    #     A.HorizontalFlip(),
    #     A.RandomBrightnessContrast(brightness_limit=0.2),
    #     A.Resize(600,600)
    # ])

    aug = A.Compose([
        A.ShiftScaleRotate(scale_limit=(0,0.1), p=0.7),
        A.RandomBrightnessContrast(brightness_limit=[-0.3,0.2], contrast_limit=[-0.3,0.1], p=1),
        A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.3),
        A.GaussNoise(var_limit=(10,50), p=0.5),
        A.CoarseDropout(p=0.3, max_holes=15, max_height=15, max_width=15),
        A.OneOf([
            A.CLAHE(p=0.7),
            A.ToGray(p=0.1),
            A.Blur(blur_limit=(5,10),p=0.2)
        ],p=1),

    ])
    max_img_num = 3000
    counts = train['label'].value_counts()
    labels = counts.index

    for label in labels:
        num_images = counts[label]
        how_many_times = (max_img_num // num_images) - 1
        manipulate_image(train, f"{label}_aug", how_many_times, aug, label)
        print()


def show_num_classes():
    import seaborn as sns
    from glob import glob
    import re

    plt.rcParams['font.family'] = 'NanumGothic'  # 원하는 폰트명으로 변경
    plt.rcParams['axes.unicode_minus'] = False

    train = pd.DataFrame({'path': glob("../../DATASET/mapping_img_data/train/*/*")})
    train['label'] = train['path'].apply(lambda x: x.split('/')[-2])

    if os.path.isdir('../../DATASET/mapping_img_data/working/'):
        train_aug = pd.DataFrame({'path': glob('../../DATASET/mapping_img_data/working/*/*')})
        train_aug['label'] = train_aug['path'].apply(lambda x: x.split('/')[-1].split('.')[0]
                                                     ).apply(lambda x: re.sub(r"[0-9]", "", x))
        #
        train = pd.concat([train, train_aug], axis=0)


    # #####################################################################################################
    # # 임의 수정
    # try:
    #     edited = train[train['label'] == '걸레받이수정']
    #     damaged = train[train['label'] == '훼손']
    #     train.drop(train[(train['label'] == '훼손') | (train['label'] == '걸레받이수정')].index, inplace=True)
    #
    #     sampled_emitted = edited.sample(n=600)
    #     sampled_dammaged = damaged.sample(n=600)
    #     del edited
    #     del damaged
    #     train = pd.concat([train, sampled_dammaged, sampled_emitted], axis=0)
    # except:
    #     print('something is empty')
    # # 임의 수정 끝!
    # ############################################################################3#######################
    counts = train['label'].value_counts()

    # countplot 생성
    plt.figure(figsize=(300,200))
    sns.set_palette('pastel')
    # sns.set_style('whitegrid')
    sns.countplot(x=train['label'], order=counts.index)

    # 그래프 제목 및 레이블 설정 (선택 사항)
    plt.title("Label Counts")
    plt.xlabel("Label")
    plt.ylabel("Count")

    # print(counts)

    # 그래프 표시
    plt.show()

if __name__ == '__main__':
    show_num_classes()
    data_augument()
    show_num_classes()