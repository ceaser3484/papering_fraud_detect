import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# load model
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '../data/dataset/cnn_model_fixed.h5') 
    if not os.path.exists(model_path):
        raise FileNotFoundError(f" 모델 파일이 존재하지 않습니다: {model_path}")

    return tf.keras.models.load_model(model_path, compile=False)

model = load_model()

def get_classes():
    dataset_path = os.path.join(os.path.dirname(__file__), '../data/dataset/train')

    if not os.path.exists(dataset_path):
        print(f" 데이터셋 폴더가 존재하지 않음: {dataset_path}")
        return []

    classes = sorted(os.listdir(dataset_path))
    if not classes:
        raise ValueError(" 데이터셋 폴더에 클래스 폴더 없음")

    return classes

classes = get_classes()

def apply_filters(img_path, filter_type="clahe"):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f" 이미지 파일 존재하지 않음: {img_path}")

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  

    if filter_type == "canny":
        processed_img = cv2.Canny(img, 50, 150)

    elif filter_type == "gaussian":
        processed_img = cv2.GaussianBlur(img, (5, 5), 0) 

    elif filter_type == "adaptive_threshold":
        processed_img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

    elif filter_type == "sobel":
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        processed_img = cv2.magnitude(sobelx, sobely)

    elif filter_type == "gabor":
        gabor_kernel = cv2.getGaborKernel((7, 7), 4.0, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        processed_img = cv2.filter2D(img, cv2.CV_8UC3, gabor_kernel)

    elif filter_type == "equalize":
        processed_img = cv2.equalizeHist(img)

    elif filter_type == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed_img = clahe.apply(img)

    else:
        raise ValueError(f" 지원되지 않는 필터 타입: {filter_type}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_rgb[:, :, ::-1]) 
    axes[0].set_title("원본 이미지 (RGB 변환됨)")
    axes[1].imshow(processed_img, cmap="gray")
    axes[1].set_title(f"적용된 필터: {filter_type}")

    for ax in axes:
        ax.axis("off")

    plt.show()

    return processed_img

# 예측
def predict_image(img_path, filter_type="clahe"):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"이미지 파일이 존재하지 않음: {img_path}")

    filtered_img = apply_filters(img_path, filter_type=filter_type)

    filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2RGB)

    filtered_img = cv2.resize(filtered_img, (224, 224))
    filtered_img = filtered_img.astype("float32") / 255.0
    filtered_img = np.expand_dims(filtered_img, axis=0)

    prediction = model.predict(filtered_img)

    print(f" 예측 확률 분포: {prediction}")

    pred_index = np.argmax(prediction)

    if pred_index >= len(classes):
        print(f" 예측된 클래스 인덱스 {pred_index}가 클래스 개수 {len(classes)}를 초과함")
        predicted_class = "알 수 없음"
    else:
        predicted_class = classes[pred_index]

    print(f"예측된 클래스: {predicted_class}")

    return predicted_class

# 실행
if __name__ == "__main__":
    test_image_path = os.path.join(os.path.dirname(__file__), "../data/sample_image.png") 

    if os.path.exists(test_image_path):
        result = predict_image(test_image_path, filter_type="clahe")
        print(f" 예측 결과: {result}")
    else:
        print(f" 이미지 파일이 존재하지 않음. 올바른 이미지 경로 확인 필요")