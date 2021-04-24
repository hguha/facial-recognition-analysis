import dlib
import cv2
import matplotlib.pyplot as plt
from imutils import face_utils
import numpy as np
from time import perf_counter, time
from config import config

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
# plt.rcParams.update({'figure.max_open_warning': 0})

# adapted from https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap='gray')
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

def eigenvalue_breakdown(show=False):
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    n_samples, h, w = lfw_people.images.shape
    X = lfw_people.data
    n_features = X.shape[1]

    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    print("Total dataset size:")
    print("n_samples: %d" % n_samples)
    print("n_features: %d" % n_features)
    print("n_classes: %d" % n_classes)

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=42)

    n_components = 150

    print("Extracting the top %d eigenfaces from %d faces"
        % (n_components, X_train.shape[0]))
    t0 = time()
    pca = PCA(n_components=n_components, svd_solver='randomized',
            whiten=True).fit(X_train)
    print("done in %0.3fs" % (time() - t0))

    eigenfaces = pca.components_.reshape((n_components, h, w))

    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("done in %0.3fs" % (time() - t0))

    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(
        SVC(kernel='rbf', class_weight='balanced'), param_grid
    )
    clf = clf.fit(X_train_pca, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)

    print("Predicting people's names on the test set")
    t0 = time()
    y_pred = clf.predict(X_test_pca)
    print("done in %0.3fs" % (time() - t0))

    print(classification_report(y_test, y_pred, target_names=target_names))
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

    prediction_titles = [title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])]

    plot_gallery(X_test, prediction_titles, h, w)

    eigenface_titles = [i for i in range(eigenfaces.shape[0])]
    plot_gallery(eigenfaces, eigenface_titles, h, w)

    plt.show()

def show_plot(image, name):
    plt.figure(figsize=(12,8))
    plt.title(name)
    plt.imshow(image, cmap='gray')
    plt.show()
    plt.close('all')


def CNN(image, name, model=config["cnn_model"], show=False):
    dnnFaceDetector = dlib.cnn_face_detection_model_v1(model)

    rects = dnnFaceDetector(image, 1)
    for (i, rect) in enumerate(rects):
        x1 = rect.rect.left()
        y1 = rect.rect.top()
        x2 = rect.rect.right()
        y2 = rect.rect.bottom()
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 3)

    if show: show_plot(image, name)

def CASCADE(image, name, model=config["haar_models"]["face"], show=False):
    cascade = cv2.CascadeClassifier(model)

    faces = cascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
        
    for (x, y, w, h) in faces: 
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 3)

    if show: show_plot(image, name)

def HOG(image, name, show=False):
    im = np.float32(image) / 255.0

    #gradient
    gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    face_detect = dlib.get_frontal_face_detector()
    rects = face_detect(image, 1)
    for rect in rects:
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 3)
        
    if show: show_plot(image, name)

if config["get_stats"]:
    print("DETECTING\n==============")

    for name, method in config["methods"].items():    
        print("Testing {} image(s) for method: {}".format(len(config["images"]), name))
        start = perf_counter()
        
        for i in config["images"]:
            image = cv2.imread(i, 0)
            locals()[method](image, name, show=config["show_plot"])
        
        print("Overall Time taken: " + str(perf_counter() - start))

        for i in config["image_types"]:
            start = perf_counter()
            
            print(i+"\n=========")
            image = cv2.imread("images/"+i+".jpg", 0)
            locals()[method](image, name)
            print("Time taken for {}: {}".format(i, str(perf_counter() - start)))
        print("\n")


if config["show_eigenvalues"]: 
    print("PREDICTING\n==============")
    eigenvalue_breakdown()