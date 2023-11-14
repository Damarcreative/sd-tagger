from __future__ import annotations

import os
import pathlib
import tarfile

import deepdanbooru as dd
import huggingface_hub
import numpy as np
import PIL.Image
import tensorflow as tf

datasets_path = input("Enter the image folder path: ")
Score_threshold = input("Score Threshold: ")

def load_sample_image_paths() -> list[pathlib.Path]:
    image_dir = pathlib.Path(datasets_path)
    return sorted(image_dir.glob('*'))


def load_model() -> tf.keras.Model:
    path = huggingface_hub.hf_hub_download('tensor-diffusion/DeepDanbooru-tagger',
                                           'model-resnet_custom_v3.h5')
    model = tf.keras.models.load_model(path)
    return model


def load_labels() -> list[str]:
    path = huggingface_hub.hf_hub_download('tensor-diffusion/DeepDanbooru-tagger',
                                           'tags.txt')
    with open(path) as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


def predict(image: PIL.Image.Image, score_threshold: float, model: tf.keras.Model, labels: list[str]) -> tuple[dict[str, float], dict[str, float], str]:
    _, height, width, _ = model.input_shape
    image = np.asarray(image)
    image = tf.image.resize(image,
                            size=(height, width),
                            method=tf.image.ResizeMethod.AREA,
                            preserve_aspect_ratio=True)
    image = image.numpy()
    image = dd.image.transform_and_pad_image(image, width, height)
    image = image / 255.
    probs = model.predict(image[None, ...])[0]
    probs = probs.astype(float)

    indices = np.argsort(probs)[::-1]
    result_all = dict()
    result_threshold = dict()
    for index in indices:
        label = labels[index]
        prob = probs[index]
        result_all[label] = prob
        if prob < score_threshold:
            break
        result_threshold[label] = prob
    result_text = ', '.join(result_all.keys())
    return result_threshold, result_all, result_text

def process_and_save_batch(image_paths: list[pathlib.Path], score_threshold: float):
    for image_path in image_paths:
      try:
            # Check if the file has an allowed extension
            if image_path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                print(f"Skipping non-image file: {image_path}")
                continue
            # Load the image
            image = PIL.Image.open(image_path)

            if image.mode == 'RGBA':
              image = image.convert('RGB')

            # Run the prediction
            result_threshold, result_all, result_text = predict(image, score_threshold, model, labels)

            result_text = result_text.replace("rating:safe, ", "")

            # Print the results
            print("Image Path:", image_path)
            print("Result Threshold:", result_threshold)
            print("Result All:", result_all)
            print("Result Text:", result_text)
            print("\n")

            # Save the result_text to a text file
            result_file_path = image_path.with_suffix('.txt')
            with open(result_file_path, 'w') as result_file:
                result_file.write(str(result_text))

      except (PIL.UnidentifiedImageError, OSError) as e:
        # Handle UnidentifiedImageError or OSError
        print(f"Error processing image {image_path}: {e}")
        continue


image_paths = load_sample_image_paths()
model = load_model()
labels = load_labels()
score_threshold = Score_threshold

process_and_save_batch(image_paths, score_threshold)
