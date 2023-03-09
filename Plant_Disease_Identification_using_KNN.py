
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 15:19:30 2022

@author: Home
"""

import cv2
import csv
import skimage

from skimage.feature import graycomatrix
from skimage.feature import graycoprops
import numpy as np
import functools  # for reduce() function
import os
from os import listdir

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Dimension of resized image
DEFAULT_IMAGE_SIZE = tuple((256, 256))

# Number of images used to train the model
N_IMAGES = 100

# Path to the dataset folder
root_dir = 'C:/Users/Asus/Desktop/'

train_dir = os.path.join(root_dir, 'images')
val_dir = os.path.join(root_dir, 'images')


def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, DEFAULT_IMAGE_SIZE)
            return image
        else:
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None


image_list, label_list = [], []

try:
    print("[INFO] Loading images ...")
    plant_disease_folder_list = listdir(train_dir)

    for plant_disease_folder in plant_disease_folder_list:
        print(f"[INFO] Processing {plant_disease_folder} ...")
        plant_disease_image_list = listdir(f"{train_dir}/{plant_disease_folder}/")

        for image in plant_disease_image_list[:N_IMAGES]:
            image_directory = f"{train_dir}/{plant_disease_folder}/{image}"
            if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                if (plant_disease_folder == 'Alternaria Alternata'):
                    image_list.append([convert_image_to_array(image_directory), 0])
                if (plant_disease_folder == 'Anthracnose'):
                    image_list.append([convert_image_to_array(image_directory), 1])
                if (plant_disease_folder == 'Bacterial Blight'):
                    image_list.append([convert_image_to_array(image_directory), 2])
                if (plant_disease_folder == 'Cercospora Leaf Spot'):
                    image_list.append([convert_image_to_array(image_directory), 3])
                if (plant_disease_folder == 'Healthy Leaves'):
                    image_list.append([convert_image_to_array(image_directory), 4])
                label_list.append(plant_disease_folder)

    print("[INFO] Image loading completed")
except Exception as e:
    print(f"Error : {e}")

# Check the number of images loaded for training
image_len = len(image_list)
print(f"Total number of images: {image_len}")


def decorrstretch(A, tol=None):
    """
    Apply decorrelation stretch to image
    Arguments:
    A   -- image in cv2/numpy.array format
    tol -- upper and lower limit of contrast stretching
    """
    # save the original shape
    orig_shape = A.shape
    # reshape the image
    #         B G R
    # pixel 1 .
    # pixel 2   .
    #  . . .      .
    A = A.reshape((-1, 3)).astype(float)
    # covariance matrix of A
    cov = np.cov(A.T)
    # source and target sigma
    sigma = np.diag(np.sqrt(cov.diagonal()))
    # eigen decomposition of covariance matrix
    eigval, V = np.linalg.eig(cov)
    # stretch matrix
    S = np.diag(1 / np.sqrt(eigval))
    # compute mean of each color
    mean = np.mean(A, axis=0)
    # substract the mean from image
    A -= mean
    # compute the transformation matrix
    T = functools.reduce(np.dot, [sigma, V, S, V.T])
    # compute offset
    offset = mean - np.dot(mean, T)
    # transform the image
    A = np.dot(A, T)
    # add the mean and offset
    A += mean + offset
    # restore original shape
    B = A.reshape(orig_shape)
    # for each color...
    for b in range(3):
        # apply contrast stretching if requested
        if tol:
            # find lower and upper limit for contrast stretching
            low, high = np.percentile(B[:, :, b], 100 * tol), np.percentile(B[:, :, b], 100 - 100 * tol)
            B[B < low] = low
            B[B > high] = high
        # ...rescale the color values to 0..255
        B[:, :, b] = 255 * (B[:, :, b] - B[:, :, b].min()) / (B[:, :, b].max() - B[:, :, b].min())
    # return it as uint8 (byte) image
    return B.astype(np.uint8)





# Main program starts
# cv2 reads image in BGR format
mainlist = []
for iiii in image_list:
    img = iiii[0]
    # cv2.imshow('original', img)

    # perform decorrelation stretching
    dcstretch = decorrstretch(img)
    # cv2.imshow('dcorrstretched', dcstretch)
    # convert into L*a*b
    labdec = cv2.cvtColor(dcstretch, cv2.COLOR_BGR2Lab)
    # cv2.imshow('LAB', labdec)

    # KNN segmentation
    K = 3
    Z = labdec.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 25)
    # cv2.imshow('Lab', lab)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((labdec.shape))
    # cv2.imshow('cluster', res2)
    print(res2.shape)
    l1 = center.tolist()
    print(l1)
    # sort the cluster centers according to the intensity of green because highest intensity value of green represents the diseased parts
    for x in range(len(l1)):
        for y in range(len(l1) - x - 1):
            if (l1[y][1] > l1[y + 1][1]):
                l1[y], l1[y + 1] = l1[y + 1], l1[y]
    print(l1)

    for x in range(res2.shape[1]):
        for y in range(res2.shape[0]):
            l2 = list(res2[y, x])
            if (l2 == l1[2]):
                res2[y, x, 0] = 0
                res2[y, x, 1] = 255
                res2[y, x, 2] = 0
            elif (l2 == l1[0]):
                res2[y, x, 0] = 0
                res2[y, x, 1] = 0
                res2[y, x, 2] = 255
            else:
                res2[y, x, 0] = 255
                res2[y, x, 1] = 0
                res2[y, x, 2] = 0
    res3 = res2.copy()
    # Green chanelling of a bgr image aims on turning the red and blue channels of desired pixels to 0 while turning the green channels to 255. This marks the pixels as absolute green.
    for i in range(res3.shape[1]):
        for j in range(res3.shape[0]):
            if (res3[j, i, 1] != 255):
                res3[j, i, 2] = 0
                res3[j, i, 0] = 0
    # Binary image.
    h = res3.shape
    w = res3.shape
    s = 0
    for i in range(h[0]):
        for j in range(w[1]):
            s += res3[i][j][1]
    # thres is the threshold for binary image conversion.
    thres = s / (h[0] * w[1])
    print(thres)
    # bin is an image array with all zeros.
    bin = np.zeros((h[0], w[1]), 'uint8')
    for i in range(h[0]):
        for j in range(w[1]):
            if res3[i][j][
                1] >= thres:  # if res3 i, j>=thres, we convert that pixel in bin to absolute white else we convert that pixel to absolute black.
                bin[i][j] = 255
            else:
                bin[i][j] = 0
    # Image erosion and reduction.
    ele = np.ones((5, 5))  # Taking an erosion structuring element.
    eimgtemp = cv2.erode(bin, ele)  # Using function to erode image.
    eimg = bin - eimgtemp  # Reducing the eroded image.

    # Final image.
    fimg = img.copy()  # Copy of the original image.
    # Taking height and width of the image.
    h = fimg.shape
    w = fimg.shape
    for i in range(h[0]):
        for j in range(w[1]):
            if (eimg[i][j] == 255):  # If there is absolute white in a pixel in reduced image, we mark it red in the final image.
                 fimg[i][j][2] = 255  # Marking red pixels in the final image.
                 fimg[i][j][0] = 0
                 fimg[i][j][1] = 0
    # keeping the disease spots and changing everything to black
    fimg1 = img.copy()  # Copy of the original image.
    # Taking height and width of the image.
    h = fimg1.shape
    w = fimg1.shape
    for i in range(h[0]):
        for j in range(w[1]):
            if (bin[i][
                j] != 255):  # If there is absolute white in a pixel in reduced image, we mark it red in the final image.
                fimg1[i][j][2] = 0  # Marking red pixels in the final image.
                fimg1[i][j][0] = 0
                fimg1[i][j][1] = 0
    # cv2.imshow('Reduced eroded image.', eimg)
    # cv2.imshow('Display', res2)
    # cv2.imshow('Green channel.', res3)
    # cv2.imshow('Binary', bin)
    # cv2.imshow('Eroded image.', eimgtemp)
    # cv2.imshow('Final image.', fimg)
    # cv2.imshow('Background black', fimg1)

    # normalized GLCM calculation from corresponding gray scale image
    gray = cv2.cvtColor(fimg1, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Gray scale', gray)
    # The MEAN and STANDARD DEVIATION are to be calculated
    # across all non-zero elements of this gray level image matrix

    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256,
                        symmetric=True)

    # due to segmentation, most of the pixels in the image are black - hence the [0,0] element of each glcm
    # will contain a very large value, whch is misleading
    # change the first row and first column  of each glcm to zero
    # before calculating the features.
    # For this reason, we have not normalized the GLCM yet.

    glcm[:, :, 0, 0][:, 0] = 0
    glcm[:, :, 0, 0][0, :] = 0
    glcm[:, :, 0, 1][:, 0] = 0
    glcm[:, :, 0, 1][0, :] = 0
    glcm[:, :, 0, 2][:, 0] = 0
    glcm[:, :, 0, 2][0, :] = 0
    glcm[:, :, 0, 3][:, 0] = 0
    glcm[:, :, 0, 3][0, :] = 0
    # print(glcm[:, :, 0, 0])

    # We shoud ideally normalize the glcm matrices now.
    # However, normalization would change the type of the matrix
    # elements from integer to float.
    # Rather than go into this typecasting problem now itself,
    # let us proceed with the integer glcm matrices.
    # We can normalize later if necessary, i.e., if it is likely
    # to improve classification results.

    # We take the mean value of each property - since four
    # values are calculated separately - one for each glcm matrix/angle
    properties = ['contrast', 'energy', 'homogeneity', 'correlation']
    contrast = graycoprops(glcm, properties[0]).mean(axis=1).astype(float)
    energy = graycoprops(glcm, properties[1]).mean(axis=1)
    energy.astype(float)
    homogeneity = graycoprops(glcm, properties[2]).mean(axis=1)
    homogeneity.astype(float)
    corr_matrix = graycoprops(glcm, properties[3])
    correlation = graycoprops(glcm, properties[3]).mean(axis=1)
    correlation.astype(float)

    # The mean and stdev are image colour properties,
    # hence those will be calculated over
    # all non-black (i.e. non-zero) pixels (these represent
    # the diseased spots on the leaf) of the gray-scale image
    gray_non_black_pixels = gray[np.nonzero(gray)]
    mean = np.mean(gray_non_black_pixels)
    stdev = np.std(gray_non_black_pixels)

    templist = []
    templist.append(iiii[1])
    templist.append(round(contrast[0], 3))
    templist.append(round(energy[0], 3))
    templist.append(round(homogeneity[0], 3))
    templist.append(round(correlation[0], 3))
    templist.append(round(mean, 3))
    templist.append(round(stdev, 3))
    mainlist.append(templist)


print(mainlist)
print("_____")
for i in mainlist:
    print("Disease category:", i[0])
    print("contrast: ", i[1])
    print("energy: ", i[2])
    print("homogeneity: ", i[3])
    print("correlation matrix: ", corr_matrix)
    print("correlation: ", i[4])
    print("Mean:", i[5])
    print("Standard deviation:", i[6])
    print("_____")
head = ['disease category', 'contrast', 'energy', 'homogeneity', 'correlation', 'mean', 'st dev']
with open('C:/Users/Asus/Desktop/dataset.csv', 'w', encoding='UTF8') as fp:
    w = csv.writer(fp)
    w.writerow(head)
    for item in mainlist:
        w.writerow(item)
    fp.close()
    print('Done')

# stop, let us check the csv file.
# os._exit()
# Let us read the feature data and target values
# from the csv file
import matplotlib.pyplot as plt

k_range = range(1, 25)

# We can create Python dictionary using [] or dict()
scores1 = []
scores = []
input_file = 'C:/Users/Asus/Desktop/dataset.csv'
X = np.genfromtxt(input_file, delimiter=',', skip_header=1, usecols=(1, 2, 3, 4, 5, 6))
y = np.genfromtxt(input_file, delimiter=',', skip_header=1, usecols=(0))

# normalize the feature values by dividing each
# attribute value by the largest value in that
# attribute column.
X_normed = X / X.max(axis=0)

# Split into training and test set

X_train, X_test, y_train, y_test = train_test_split(

    X_normed, y, test_size=0.2, random_state=42)
best_acc=-1.
best_k=-1
conf_m=0.
for i in k_range:
    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, y_train)

    # Predict on dataset which model has not seen before

    y_pred = knn.predict(X_test)
    y_pred1 = knn.predict(X_train)
    scores.append(accuracy_score(y_test, y_pred))
    if(accuracy_score(y_test, y_pred)>best_acc):
        best_acc=accuracy_score(y_test, y_pred)
        best_k=i
        conf_m=confusion_matrix(y_test, y_pred)
    scores1.append(accuracy_score(y_train, y_pred1))
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    plt.title('k-NN: Varying Number of Neighbors')
plt.plot(k_range, scores, label='Testing Accuracy')
plt.plot(k_range, scores1, label='Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
text='Highest accuracy is {}'
plt.title(text.format(best_acc))
plt.show()
plt.close()
conf_disp=ConfusionMatrixDisplay(confusion_matrix=conf_m, display_labels=['AA','A','B','C','H'])
conf_disp.plot()
plt.show()
plt.close()
print("Accuracy score=",(best_acc*100.))
print("K value=",best_k)