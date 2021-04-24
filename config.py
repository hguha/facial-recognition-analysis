from os import listdir
from os.path import isfile, join

images_folder = "images/"
images = [images_folder+f for f in listdir(images_folder) if isfile(join(images_folder, f))]

config = {
    "images": images,
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
        "Convolutional Neural Network": "CNN", 
        "Histogram of Oriented Gradients": "HOG", 
        "Cascade Classifer": "CASCADE"
    },

    "image_types": ["group", "low-light", "obstructed", "single", "angle", "small"]

}