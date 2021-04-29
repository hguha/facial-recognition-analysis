from os import listdir, walk
from os.path import isfile, join, isdir

images_folder = "images/"
image_categories = [f for f in listdir(images_folder) if isdir(join(images_folder, f))]
images = dict()
for i in image_categories:
    images[i] = [images_folder+i+"/"+f for f in listdir(images_folder+i+"/")]

config = {
    "images": images,
    "images_flat": [item for sublist in images.values() for item in sublist],
    "show_plot": False,
    "show_eigenvalues": False,
    "get_stats": True,
    "cnn_model":"models/mmod_human_face_detector.dat",

    "haar_models": {
        "face": "models/haarcascade_frontalface_default.xml",
        "eyes": "models/haarcascade_eye.xml",
        "mouth": "models/haarcascade_smile.xml"
    },

    "methods": {
        # "Convolutional Neural Network": "CNN", 
        # "Histogram of Oriented Gradients": "HOG", 
        "Cascade Classifer": "CASCADE"
    },

    "image_types": image_categories

}