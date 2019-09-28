from flask import Flask, render_template, request
import os
import time
import numpy as np
import time
import sys
import requests
import cv2
from sklearn.cluster import KMeans
from collections import Counter
import imutils
import pprint
from matplotlib import pyplot as plt

FD_API_KEY=os.environ['FD_API_KEY']
FD_API_SECRET=os.environ['FD_API_SECRET']
UPLOAD_DIR='files'

app = Flask(__name__)

def cropImg(path, x, y, w, h):
    img = cv2.imread(path)
    if (img is None):
        print("Can't open image file")
        return 0
    r = max(w, h) / 2
    centerx = x + w / 2
    centery = y + h / 2
    nx = int(centerx - r)
    ny = int(centery - r)
    nr = int(r * 2)
    faceimg = img[ny:ny + nr, nx:nx + nr]
    return faceimg

def cropMouthImg(path, x, y, w, h):
    img = cv2.imread(path)
    if (img is None):
        print("Can't open image file")
        return 0
    r = max(w, h) / 2
    centerx = x + r
    centery = y + r
    nx = int(centerx - r)
    ny = int(centery - r)
    nr = int(r * 2)
    faceimg = img[ny:ny + int(h), nx:nx + nr]
    return faceimg

def getFaceImage(path):
    # headers = {'Content-Type': 'multipart/form-data'}
    data = {
        'api_key': FD_API_KEY,
        'api_secret': FD_API_SECRET,
        'return_landmark': 1,
        'return_attributes': 'gender,age,mouthstatus,skinstatus',
    }
    files = {'image_file': open(path, 'rb')}
    response = requests.post('https://api-us.faceplusplus.com/facepp/v3/detect', data=data, files=files)
    if response.ok:
        body = response.json()
        face =body['faces'][0]['face_rectangle']
        lm=body['faces'][0]['landmark']
        left=lm['mouth_left_corner']
        right=lm['mouth_right_corner']
        top=lm['mouth_upper_lip_top']
        bottom=lm['mouth_lower_lip_bottom']
        w = face['width']
        h = face['height']
        x = face['left']
        y = face['top']
        m_w = abs(right['x'] - left['x'])
        m_h = abs(top['y'] - bottom['y'])
        m_x = left['x']
        m_y = top['y']
        face=cropImg(path,x,y,w,h)
        mouth=cropMouthImg(path,m_x,m_y,m_w,m_h)
        timestr = time.strftime("%Y%m%d%H%M%S%MS")

        filePathFace=f'{UPLOAD_DIR}/face-{timestr}.jpg'
        cv2.imwrite(filePathFace, face)

        filePathMouth = f'{UPLOAD_DIR}/mouth-{timestr}.jpg'
        cv2.imwrite(filePathMouth, mouth)

        return {'face': filePathFace, 'mouth': filePathMouth}
    else:
        return -1

def extractSkin(image):
    # Taking a copy of the image
    img = image.copy()
    # Converting from BGR Colours Space to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Defining HSV Threadholds
    lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
    upper_threshold = np.array([20, 255, 255], dtype=np.uint8)

    # Single Channel mask,denoting presence of colours in the about threshold
    skinMask = cv2.inRange(img, lower_threshold, upper_threshold)

    # Cleaning up mask using Gaussian Filter
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    # Extracting skin from the threshold mask
    skin = cv2.bitwise_and(img, img, mask=skinMask)

    # Return the Skin image
    return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)

def extractSkiTeath(image):
    # Taking a copy of the image
    img = image.copy()
    # Converting from BGR Colours Space to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Defining HSV Threadholds
    lower_threshold = np.array([10, 10, 0], dtype=np.uint8)
    upper_threshold = np.array([200, 200, 200], dtype=np.uint8)

    # Single Channel mask,denoting presence of colours in the about threshold
    skinMask = cv2.inRange(img, lower_threshold, upper_threshold)

    # Cleaning up mask using Gaussian Filter
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    # Extracting skin from the threshold mask
    skin = cv2.bitwise_and(img, img, mask=skinMask)

    # Return the Skin image
    return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)


def removeBlack(estimator_labels, estimator_cluster):

    # Check for black
    hasBlack = False

    # Get the total number of occurance for each color
    occurance_counter = Counter(estimator_labels)

    # Quick lambda function to compare to lists
    def compare(x, y): return Counter(x) == Counter(y)

    # Loop through the most common occuring color
    for x in occurance_counter.most_common(len(estimator_cluster)):

        # Quick List comprehension to convert each of RBG Numbers to int
        color = [int(i) for i in estimator_cluster[x[0]].tolist()]

        # Check if the color is [0,0,0] that if it is black
        if compare(color, [0, 0, 0]) == True:
            # delete the occurance
            del occurance_counter[x[0]]
            # remove the cluster
            hasBlack = True
            estimator_cluster = np.delete(estimator_cluster, x[0], 0)
            break

    return (occurance_counter, estimator_cluster, hasBlack)


def getColorInformation(estimator_labels, estimator_cluster, hasThresholding=False):

    # Variable to keep count of the occurance of each color predicted
    occurance_counter = None

    # Output list variable to return
    colorInformation = []

    # Check for Black
    hasBlack = False

    # If a mask has be applied, remove th black
    if hasThresholding == True:

        (occurance, cluster, black) = removeBlack(
            estimator_labels, estimator_cluster)
        occurance_counter = occurance
        estimator_cluster = cluster
        hasBlack = black

    else:
        occurance_counter = Counter(estimator_labels)

    # Get the total sum of all the predicted occurances
    totalOccurance = sum(occurance_counter.values())

    # Loop through all the predicted colors
    for x in occurance_counter.most_common(len(estimator_cluster)):

        index = (int(x[0]))

        # Quick fix for index out of bound when there is no threshold
        index = (index-1) if ((hasThresholding & hasBlack)
                              & (int(index) != 0)) else index

        # Get the color number into a list
        color = estimator_cluster[index].tolist()

        # Get the percentage of each color
        color_percentage = (x[1]/totalOccurance)

        # make the dictionay of the information
        colorInfo = {"cluster_index": index, "color": color,
                     "color_percentage": color_percentage}

        # Add the dictionary to the list
        colorInformation.append(colorInfo)

    return colorInformation


def extractDominantColor(image, number_of_colors=5, hasThresholding=False):

    # Quick Fix Increase cluster counter to neglect the black(Read Article)
    if hasThresholding == True:
        number_of_colors += 1

    # Taking Copy of the image
    img = image.copy()

    # Convert Image into RGB Colours Space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Reshape Image
    img = img.reshape((img.shape[0]*img.shape[1]), 3)

    # Initiate KMeans Object
    estimator = KMeans(n_clusters=number_of_colors, random_state=0)

    # Fit the image
    estimator.fit(img)

    # Get Colour Information
    colorInformation = getColorInformation(
        estimator.labels_, estimator.cluster_centers_, hasThresholding)
    return colorInformation

def getDominantColor(imagePath):
    imgPath=getFaceImage(imagePath)
    if imgPath == -1:
        print('Error ImpPath')
        sys.exit(-1)
    face=cv2.imread(imgPath['face'])
    faceSkin = extractSkin(face)
    faceColors=extractDominantColor(faceSkin, hasThresholding=True)

    mouth = cv2.imread(imgPath['mouth'])
    mouthSkin = extractSkin(mouth)
    mouthColors = extractDominantColor(mouthSkin, hasThresholding=True)

    return {'mouth':mouthColors, 'face': faceColors}

def skinRedness(colors):
    maxProb=0
    for colorRange in colors:
        normalizedColors = [0,0,0]
        for i in range(0, 3):
            normalizedColors[i] = colorRange['color'][i] / 255
        red = normalizedColors[0]
        prob =  max(0, red - max(normalizedColors[1], normalizedColors[2]))
        maxProb=max(maxProb, prob)
    return maxProb

def toothPlaque(colors):
    maxProb = 0
    for colorRange in colors:
        normalizedColors = [0, 0, 0]
        for i in range(0, 3):
            normalizedColors[i] = colorRange['color'][i] / 255
        red = normalizedColors[0]
        green = normalizedColors[1]
        blue = normalizedColors[2]
        if abs(red - green) <= 0.2:
            prob = max(0, (red + green)/2 - blue)
            maxProb = max(maxProb, prob)
    return maxProb


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        timestr = time.strftime("%Y%m%d%H%M%S%MS")
        filename, ext = os.path.splitext(f.filename)
        path = f'{UPLOAD_DIR}/file-{timestr}{ext}'
        f.save(path)
        colors = getDominantColor(path)
        return {
            'face_redness_percentage': int(skinRedness(colors['face'])*100),
            'teeth_plaque_percentage': int(toothPlaque(colors['mouth'])*100)
        }

if __name__ == '__main__':
    app.run(host= '0.0.0.0')