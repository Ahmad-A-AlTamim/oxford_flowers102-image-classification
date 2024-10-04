import matplotlib
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import tf_keras
import numpy as np
tfds.disable_progress_bar()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
import json
from PIL import Image
import argparse

def load_model(model_path):
    loaded_model = tf_keras.models.load_model(
        model_path,
        custom_objects={'KerasLayer': hub.KerasLayer}  # Specify the custom layer
    )
    return loaded_model

def process_image(image_array):
    image_tensor = tf.convert_to_tensor(np.array(image_array))
    image_tensor = tf.cast(image_tensor, tf.float32)
    image_tensor /= 255
    image_tensor = tf.image.resize(image_tensor, [224, 224])
    return image_tensor.numpy()


def predict(image_path, model, top_k):
    from PIL import Image

    img = Image.open(image_path)
    test_image = np.asarray(img)

    processed_image = process_image(test_image)
    processed_image = np.expand_dims(processed_image, axis=0)

    model = load_model(args.model_path)
    img_prob = model.predict(processed_image)

    return -np.sort(-img_prob)[:, :int(top_k)], np.argsort(-img_prob)[:, :int(top_k)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict image class using a Keras model.')
    parser.add_argument('image_path', type=str, help='Path to the image file.')
    parser.add_argument('model_path', type=str, help='Path to the Keras model file.')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K predictions (default: 5).')
    parser.add_argument('--class_names', type=str, default='./label_map.json', help='Path to JSON file containing class names.')

    args = parser.parse_args()
    with open(args.class_names, 'r') as f:
        class_names = json.load(f)
    probs,classes= predict(args.image_path, args.model_path, args.top_k)
    classes = [class_names[str(classes.squeeze()[i] + 1)] for i in range(args.top_k)]
    try:
        os.system('cls')
    except:
        pass
    print('\n\n\nResults:')
    for i in range(args.top_k):
        print(f'Name : {i+1}. {classes[i]}|| Probabilities : {probs.squeeze()[i] * 100:.2f}%')
