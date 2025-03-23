import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from PIL import Image

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../../data/cnn_model_gradcam.h5')
TEST_IMAGE_PATH = os.path.join(BASE_DIR, '../../data/sample_image.jpeg')

# 클래스 목록
CLASSES = [
    '가구수정', '걸레받이수정', '곰팡이', '꼬임', '녹오염', '들뜸', '면불량', '몰딩수정', '반점', '석고수정',
    '오염', '오타공', '울음', '이음부불량', '창문,문틀수정', '터짐', '틈새과다', '피스', '훼손'
]

def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

def preprocess_input(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img).astype('float32') / 255.0
    return np.expand_dims(img, axis=0), img

def make_gradcam_heatmap(model, img_array, last_conv_layer_name="conv4_block6_out"):
    _ = model(img_array)  # 모델 구조 확립

    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        raise ValueError(f"{last_conv_layer_name} 레이어를 찾을 수 없습니다.")

    grad_model = Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), int(pred_index)

def display_gradcam(img_path, heatmap, pred_class, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Prediction: {pred_class}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# 실행
if __name__ == '__main__':
    model = load_model()
    img_input, _ = preprocess_input(TEST_IMAGE_PATH)

    heatmap, pred_index = make_gradcam_heatmap(model, img_input)
    pred_class = CLASSES[pred_index]

    display_gradcam(TEST_IMAGE_PATH, heatmap, pred_class)