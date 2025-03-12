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
    manipulate_image(train, 'train_301', 200, aug,
                     '반점')
    print()
    manipulate_image(train, 'train_302', 115, aug,
                     '틈새과다')
    print()
    manipulate_image(train, 'train_303', 50, aug,
                     '가구수정')
    print()
    manipulate_image(train, 'train_304', 40, aug,
                     '녹오염')
    print()
    manipulate_image(train, 'train_305', 35, aug,
                     '이음부불량')
    print()
    manipulate_image(train, 'train_306', 26, aug,
                     '울음')
    print()
    manipulate_image(train, 'train_307', 20, aug,
                     '창틀,문틀수정')
    print()
    manipulate_image(train, 'train_308', 10, aug,
                     '피스')
    print()
    manipulate_image(train, 'train_309', 10, aug,
                     '들뜸')
    print()
    manipulate_image(train, 'train_310', 9, aug,
                     '석고수정')

    print()
    manipulate_image(train, 'train_311', 5, aug,
                     '면불량')
    print()
    manipulate_image(train, 'train_312', 4, aug,
                     '몰딩수정')
    print()
    manipulate_image(train, 'train_313', 6, aug,
                     '오타공','곰팡이')
    print()
    manipulate_image(train, 'train_313', 5, aug,
                     '터짐')
    print()
    manipulate_image(train, 'train_313', 4, aug,
                     '꼬임')
    print()
    manipulate_image(train, 'train_313', 2, aug,
                     '걸레받이수정')


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


    #####################################################################################################
    # 임의 수정
    try:
        edited = train[train['label'] == '걸레받이수정']
        damaged = train[train['label'] == '훼손']
        train.drop(train[(train['label'] == '훼손') | (train['label'] == '걸레받이수정')].index, inplace=True)

        sampled_emitted = edited.sample(n=600)
        sampled_dammaged = damaged.sample(n=600)
        del edited
        del damaged
        train = pd.concat([train, sampled_dammaged, sampled_emitted], axis=0)
    except:
        print('something is empty')
    # 임의 수정 끝!
    ############################################################################3#######################
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

    # 그래프 표시
    plt.show()

if __name__ == '__main__':
    show_num_classes()
    # data_augument()
    # show_num_classes()