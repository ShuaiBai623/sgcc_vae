import os
from os import path
import sys

folder_path = "/data/fhz/unsupervised_recommendation_model"
if not path.exists(folder_path):
    os.makedirs(folder_path)
    os.makedirs(path.join(folder_path, "dataset"))
    os.makedirs(path.join(folder_path, "model_parameter", "vae"))
    os.makedirs(path.join(folder_path, "model_parameter", "idfe"))

lib_path = path.abspath(path.dirname(__file__))
if lib_path not in sys.path:
    sys.path.insert(lib_path, 0)
