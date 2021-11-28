# Prerequisites:
# Tensorflow 2.0+ (pip install --upgrade tensorflow)
# PIL (Built-in to Python)
# Tensorflow Hub (pip install --upgrade tensorflow-hub)

# AUTHOR: Aryan Mishra (https://www.github.com/ahmishra)

# GOOGLE COLAB NOTEBOOK: https://www.shorturl.at/kqzE2
# GITHUB: https://github.com/ahmishra/CelebA-ProGAN

from time import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
start = time()
verbose = True

if verbose:
    print("[INFO] Loading libraries")

from tensorflow.image import convert_image_dtype
from tensorflow.random import normal
from tensorflow import constant
from tensorflow_hub import load
from tensorflow import Variable
from tensorflow import uint8

from matplotlib.pyplot import imshow, show, tick_params
from PIL.Image import fromarray

if verbose:
    print("[INFO] Loading helpers")

def display_image(image):
  image = constant(image)
  image = convert_image_dtype(image, uint8)
  tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
  imshow(image.numpy())
  show()

if verbose:
    print("[GEN] Generating face...")

latent_dim = 512
progan = load("https://tfhub.dev/google/progan-128/1").signatures['default']
initial_vector = normal([1, latent_dim])
vector = Variable(initial_vector)
image = progan(vector.read_value())['default'][0]

if verbose:
    print("[INFO] Displaying generated face...")
    print(f"Finished in {round(time() - start, 2)}s")
    
display_image(image)
