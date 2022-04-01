"""
        '########:'##::::'##::::'##:::
         # .....::. ##::'##:::'####:::
         # ::::::::. ##'##::::.. ##:::
         # :::::. ###::::::: ##:::
         # ...:::::: ## ##:::::: ##:::
         # :::::::: ##:. ##::::: ##:::
         # : ##:::. ##::'######:
        ........::..:::::..:::......::
"""
import math
from typing import List, Tuple
from cv2 import IMREAD_COLOR, IMREAD_GRAYSCALE

import numpy as np
import cv2
import matplotlib.pyplot as plt
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 316071349


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE(1) or RGB(2)
    :return: The image object
    """
    representation = IMREAD_COLOR if representation == 2 else IMREAD_GRAYSCALE
    image = cv2.imread(filename, representation).astype(np.float32)/255
    if representation == IMREAD_COLOR:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    image = imReadAndConvert(filename, representation)
    image = (image * 255).astype(np.uint8)
    plt.gray()
    plt.imshow(image)
    plt.show(block=True)


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    CONVERSION_MATRIX = np.array([[0.299,      0.587,            0.114],
                                 [0.59590059, -0.27455667, -0.32134392],
                                 [0.21153661, -0.52273617,  0.31119955]])
    res = np.dot(imgRGB, CONVERSION_MATRIX.T.copy())
    return res


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    CONVERSION_MATRIX = np.array([[1,  0.95598634157313524654,  0.62082479797875664386],
                                  [1, -0.27201283148102584175, -
                                      0.64720424184610618459],
                                  [1, -1.10674021097373043540,  1.70423048568435170030]])
    res = imgYIQ.dot(CONVERSION_MATRIX.T.copy())

    return res


def hsitogramEqualize(imgOrig: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """

    grayscale = True
    img = imgOrig
    # img is float

    if len(img.shape) == 3:
        grayscale = False
        imgOrig = transformRGB2YIQ(imgOrig)
        img = imgOrig[:, :, 0]

    img = (img * 255).astype(np.uint8)
    # img is int
    hist = np.histogram(img, bins=256, range=(0, 255))[0]
    cumsum = np.cumsum(hist)
    pixle_count = img.size
    lut = np.array([math.ceil((s/pixle_count)*255) for s in cumsum])
    eq_img = np.array([[lut[p] for p in row] for row in img])
    eq_hist = np.histogram(eq_img, bins=256, range=(0, 255))[0]
    eq_img = eq_img.astype(np.float64)/255
    # eq_img is float
    if not grayscale:
        imgOrig[:, :, 0] = eq_img
        eq_img = transformYIQ2RGB(imgOrig)
    return eq_img, hist, eq_hist


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> Tuple[List[np.ndarray], List[float]]:
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    grayscale = len(imOrig.shape) < 3

    img_copy = np.array(imOrig)
    img = img_copy
    if not grayscale:
        img_copy = transformRGB2YIQ(img_copy)
        img = img_copy[:, :, 0]
    img = (img * 255).astype(np.uint8)
    q_images = []
    errors = []
    z = np.linspace(0, 255, num=nQuant + 1, dtype=np.uint8)
    hist, bins = np.histogram(img, bins=255)
    for i in range(nIter):
        w = [hist[z[j]:z[j+1]] for j in range(nQuant)]
        q = np.array([np.average(range(z[j], z[j+1]), weights=w[j] if sum(w[j])
                     > 0 else None) for j in range(nQuant)], dtype=np.float128)
        q_image = np.array(img)
        for j in range(nQuant):
            condition = (z[j] <= img) & (img <= z[j+1])
            if math.isnan(q[j]):
                q[j] = 0
            q_image[condition] = int(q[j])

        q_image = q_image.astype(float)/255
        if not grayscale:
            img_copy[:, :, 0] = q_image
            q_image = transformYIQ2RGB(img_copy)

        q_images.append(q_image)
        error = np.sqrt(
            np.sum(np.power(imOrig - q_image, 2)))/imOrig.size
        errors.append(error)
        z = [0]+[int((q[j]+q[j+1])/2) for j in range(nQuant-1)]+[255]
    # if grayscale:
    #     plt.imshow(q_image, cmap='gray')
    # else:
    #     plt.imshow(q_image)
    # plt.show()
    return q_images, errors
