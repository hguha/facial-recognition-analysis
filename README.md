# Getting Started
To run, simply run the following commands:
```
python3 -m pip install -r requirements.txt #if using a mac, create a venv as necessary
python3 sol.py
```

# Configuration

In `config.py`, specify the following parameters of how you'd like the program to run when you type `python3 sol.py`. Here are some important configs:

```
    "large_images_folder" : specify the folder that will contain your image dataset. 
    "large_image_bbox" : specify the folder that will contain your image dataset labels. 
    "show_plot": True/False on whether to show a plot with the image and detected bounding box,
    "show_eigenvalues": True/False whether to run the PCA/SVM analysis,
    "get_stats": True/False whether to show the accuracy and time statistics,
```