# The saved .h5 model has been trained on TensorFlow-2.6.0, so this script should be run under an environment with the same TensorFlow version.

import tensorflow as tf
import tensorflow_hub as hub

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore')

import numpy as np

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

import json

from PIL import Image

import argparse

parser = argparse.ArgumentParser(
    description = 'Parsing arguments for predict.py',
)
parser.add_argument('image_path', action = "store", type = str)
parser.add_argument('model', action = "store", type = str)
parser.add_argument('--top_k', action = "store", default = 3, dest = "top_k", type = int)
parser.add_argument('--category_names', action = "store", default = './label_map.json', dest = "label_map", type = str)

results = parser.parse_args()

print(results.image_path)
print(results.model)


def process_image(image_array):
	image_tensor = tf.convert_to_tensor(np.array(image_array))
	image_tensor = tf.cast(image_tensor, tf.float32)
	image_tensor /= 255
	image_tensor = tf.image.resize(image_tensor, [224, 224])
	return image_tensor.numpy()


with open(results.label_map, 'r') as f:
    class_names = json.load(f)

	
img = Image.open(str(results.image_path))
test_image = np.asarray(img)

processed_image = process_image(test_image)
processed_image = np.expand_dims(processed_image, axis = 0)


reloaded_model = tf.keras.models.load_model(str(results.model), 
                                            custom_objects = {'KerasLayer': hub.KerasLayer})

img_prob = reloaded_model.predict(processed_image)

return_probs = -np.sort(-img_prob)[:, :int(results.top_k)]
return_classes = np.argsort(-img_prob)[:, :int(results.top_k)]

classes = [class_names[str(return_classes.squeeze()[i] + 1)] for i in range(results.top_k)]

print('\n\n\nResults:')
print(f'Top {results.top_k} most likely Probabilities for the given image: ', return_probs.squeeze())
print(f'Top {results.top_k} most likely Classes/Flower Names for the given image: ', classes)

