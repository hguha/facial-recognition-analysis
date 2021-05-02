import dlib
import cv2
import matplotlib.pyplot as plt
from imutils import face_utils
import numpy as np
from time import perf_counter, time
from config import config
import csv
from os.path import join
import math

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

    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("done in %0.3fs" % (time() - t0))

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
    global dnnFaceDetector

    bboxes = []

    rects = dnnFaceDetector(image, 1)
    for (i, rect) in enumerate(rects):
        x1 = rect.rect.left()
        y1 = rect.rect.top()
        x2 = rect.rect.right()
        y2 = rect.rect.bottom()
        bboxes.append((x1, y1, x2, y2))
        if show: cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 3)

    if show: show_plot(image, name)

    return bboxes



def CASCADE(image, name, model=config["haar_models"]["face"], show=False):
    global cascade

    faces = cascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    bboxes = []
        
    for (x, y, w, h) in faces: 
        bboxes.append((x, y, x + w, y + h))
        if show: cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 3)

    if show: show_plot(image, name)

    return bboxes

def HOG(image, name, show=False):
    global face_detect
    im = np.float32(image) / 255.0

    #gradient
    gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    
    rects = face_detect(image, 1)
    bboxes = []
    for rect in rects:
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        bboxes.append((x, y, x + w, y + h))
        if show: cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 3)
        
    if show: show_plot(image, name)
    return bboxes

#euclidean distance
def eucl_dist(x1, x2, y1, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

#Find the closest box for each box in boxes1. We use the min of the distance between the upper left and lower right
#corner, since the generated box is often smaller than the label, and it is close to one corner or the other
def findClosest(boxes1, boxes2):
    closestFound = {}
    for box in boxes1:
        (x1, y1, x2, y2) = box
        closest = None
        dist = 100000
        for box2 in boxes2:
            (x11, y11, x22, y22) = box2
            euc1 = eucl_dist(x1, x11, y1, y11)
            euc2 = eucl_dist(x2, x22, y2, y22)
            if min(euc1, euc2) < dist:
                closest = box2
                dist = min(euc1, euc2)
        closestFound[box] = (closest, dist)
    return closestFound

def add_tuple(x, y):
    (a, b, c) = x
    (d, e, f) = y
    return (a + d, b + e, c + f)

#The reference boxes and the found boxes - we want to determine which are correct, which are FP, and which are FN
def evalBoxes(boxes1, boxes2):
    #we will find the closest found box for each real box, then sort by distance (because there can be false positive and negatives)
    #first, we find the closest real box for each found box
    correctMatch = 0
    falsePos = 0
    falseNeg = 0

    b1 = [i for i in boxes1]
    b2 = [i for i in boxes2]
    while len(b1) > 0 and len(b2) > 0:
        close = findClosest(b1, b2)
        (boxR, x) = min(close.items(), key= lambda x : x[1][1])
        (boxF, dist) = x
        if (dist < 50):
            correctMatch += 1
            print(boxF)
            b1.remove(boxR)
            b2.remove(boxF)
        else:
            break
    #Now everything left in b1 is is a false negative, and everything left in b2 is a false positive
    falsePos = len(b2)
    falseNeg = len(b1)
    print(correctMatch, falsePos, falseNeg)
    return (correctMatch, falsePos, falseNeg)


#Does automated analysis of the images, since the dataset is labelled. Currently, you have to
#comment out the others and choose a model. We could make this nicer.
if config["large_image_set"]:
    bboxes = {}

    with open(config["large_image_bbox"]) as txtfile:
        while True:
            filename = txtfile.readline().rstrip()
            if not filename:
                break
            numFaces = int(txtfile.readline())
            faces = []
            for i in range(numFaces):
                face = txtfile.readline().split()
                majRad = int(float(face[0]))
                minRad = int(float(face[1]))
                cenX = int(float(face[3]))
                cenY = int(float(face[4]))
                x1 = cenX - minRad
                y1 = cenY - majRad
                x2 = cenX + minRad
                y2 = cenY + majRad
                faces.append((x1, y1, x2, y2))
            bboxes[filename] = faces
    print(bboxes)

    #dnnFaceDetector = dlib.cnn_face_detection_model_v1(config["cnn_model"])
    cascade = cv2.CascadeClassifier(config["haar_models"]["face"])
    #face_detect = dlib.get_frontal_face_detector()

    oneFace = (0, 0, 0)
    twoFace = (0, 0, 0)
    moreFace =(0, 0, 0)
    numOneFace = 0
    numTwoFace = 0
    numMoreFace = 0
    ctr = 0
    start = perf_counter()
    for filename in bboxes.keys():
        print(ctr)
        ctr += 1
        print(join(config["large_images_folder"], filename) + ".jpg")
        image = cv2.imread((join(config["large_images_folder"], filename) + ".jpg"), 0)
        #bbox_found = CNN(image, "")
        bbox_found= CASCADE(image, "")
        #bbox_found = HOG(image, "")
        faces = bboxes[filename]

        result =  evalBoxes(faces, bbox_found)
        (correct, falsePos, falseNeg) = result
        numFaces = correct + falseNeg
        if numFaces == 1:
            numOneFace += 1
            oneFace = add_tuple(oneFace, result)
        elif numFaces == 2:
            numTwoFace += 1
            twoFace = add_tuple(twoFace, result)
        elif numFaces > 0:
            numMoreFace += 1
            moreFace = add_tuple(moreFace, result)
    print ("There were " + str(numOneFace) + " images with 1 face")
    print(oneFace)
    print ("There were " + str(numTwoFace) + " images with 2 faces")
    print(twoFace)
    print ("There were " + str(numMoreFace) + " images with > 2 faces")
    print(moreFace)
    print("TIME TAKEN:")
    print(str(perf_counter() - start))



if config["get_stats"]:
    print("DETECTING\n==============")

    for name, method in config["methods"].items():    
        print("Testing {} image(s) for method: {}".format(len(config["images_flat"]), name))
        start = perf_counter()
        
        for i in config["images_flat"]:
            image = cv2.imread(i, 0)
            locals()[method](image, name, show=config["show_plot"])
        
        print("Overall Time taken: " + str(perf_counter() - start))

        for i in config["image_types"]:
            start = perf_counter()
            
            print(i+"\n=========")
            for img in config["images"][i]:
                image = cv2.imread(img, 0)
                locals()[method](image, name)
            
            n = len(config["images"][i])
            print("Average Time of {} images for {}: {}".format(n, i, str((perf_counter() - start)/n)))
        print("\n")


if config["show_eigenvalues"]: 
    print("PREDICTING\n==============")
    eigenvalue_breakdown()