import tensorflow as tf;
from rembg import remove
from PIL import Image
import numpy as np
import requests

new_model = tf.keras.models.load_model("model.keras")
x_mean = 72.94035223214286
x_std = 90.02118235130519
epsilon = 1e-10

classes = [
    "top", 
    "pant",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt", 
    "sneaker",
    "bag", 
    "ankle boot"
]
def predict_image(img):
    input = Image.open(requests.get(img, stream=True).raw) # load image
    output = remove(input) # remove background
    output = output.resize((28, 28))
    output = output.convert('L')
    output = np.array(output) # convert to numpy array
    output = output.reshape(1, 784) # reshape to 1x784
    output = (output - x_mean) / (x_std + epsilon) # normalize
    output = np.argmax(new_model.predict(output)) # predict label
    return classes[output] 