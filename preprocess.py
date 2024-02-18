import tensorflow as tf
from tensorflow.python.keras.layers import Concatenate
import os
import matplotlib.pyplot as plt

def preprocess_image(image_paths: list[str]) -> list[float]:
    
    processed_images = []
    
    for image_path in image_paths:
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [406, 612])
        img = img / 255.0
        processed_images.append(img)
    
    return processed_images

def get_preprocessed(dogs_path, bagels_path):
    os.chdir(dogs_path)

    image_paths = [file for file in os.listdir()]
    processed_dog_images = preprocess_image(image_paths)
    dog_labels = [1] * len(processed_dog_images)
    # print(processed_dog_images)

    # preprocessing bagels

    os.chdir(bagels_path)

    image_paths = [file for file in os.listdir()]
    processed_bagel_images = preprocess_image(image_paths)
    bagel_labels = [0] * len(processed_bagel_images)
    
    return [[processed_bagel_images, processed_dog_images], [bagel_labels, dog_labels]]
