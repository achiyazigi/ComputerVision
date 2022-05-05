import math
import sys
from typing import List, Tuple

import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt


def myID() -> np.uint8:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 316071349

# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10,
                win_size=5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    half_win_size = win_size / 2.0
    floor = math.floor(half_win_size)
    res = []
    pts = []
    if im1.ndim > 2:
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    im1 = cv2.copyMakeBorder(im1, floor, floor, floor,
                             floor, borderType=cv2.BORDER_REPLICATE)
    im2 = cv2.copyMakeBorder(im2, floor, floor, floor,
                             floor, borderType=cv2.BORDER_REPLICATE)

    Ix = cv2.Sobel(im2, cv2.CV_64F, 1, 0, ksize=3,
                   borderType=cv2.BORDER_DEFAULT)
    Iy = cv2.Sobel(im2, cv2.CV_64F, 0, 1, ksize=3,
                   borderType=cv2.BORDER_DEFAULT)

    # plt.subplot(1,2,1),plt.imshow(Ix,cmap = 'gray')
    # plt.title('x'), plt.xticks([]), plt.yticks([])
    # plt.subplot(1,2,2),plt.imshow(Iy,cmap = 'gray')
    # plt.title('y'), plt.xticks([]), plt.yticks([])
    # plt.show
    for y in range(floor, Ix.shape[0] - floor-1, step_size):
        for x in range(floor, Ix.shape[1] - floor-1, step_size):
            It: np.ndarray = im1[y - floor:y+floor+1, x-floor:x +
                                 floor+1] - im2[y - floor:y+floor+1, x-floor:x+floor+1]
            nix: np.ndarray = Ix[y - floor:y+floor+1, x-floor:x+floor+1]
            niy: np.ndarray = Iy[y - floor:y+floor+1, x-floor:x+floor+1]
            A: np.ndarray = np.concatenate(
                [np.atleast_2d(nix.flatten()).T, np.atleast_2d(niy.flatten()).T], axis=1)
            mat = A.T @ A
            eigvals = np.linalg.eigvalsh(mat)
            if eigvals.size > 1 and eigvals[-2] > 1 and eigvals[-1]/eigvals[-2] < 100:
                mat_inverse = np.linalg.inv(mat)
                uv = mat_inverse @ A.T @ np.atleast_2d(It.flatten()).T
                res.append(uv.T.reshape(-1))
                pts.append(np.array([x-floor, y-floor]))
    return np.array(pts), np.array(res)


def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int,
                     stepSize: int, winSize: int) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """

    if img1.ndim > 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    p = []
    uvs = np.zeros((*img2.shape, 2))
    for _ in range(k):
        p.append(np.array([img1.copy(), img2.copy()]))
        img1 = cv2.pyrDown(img1, dstsize=(img1.shape[1]//2, img1.shape[0]//2))
        img2 = cv2.pyrDown(img2, dstsize=(img2.shape[1]//2, img2.shape[0]//2))
    for level in range(k-1, -1, -1):
        pyr1, pyr2 = p[level]
        pts, uv = opticalFlow(pyr1, pyr2, max(
            int(stepSize * math.pow(2, -level)), 1), winSize)
        converted_points = pts * np.power(2, level)
        try:
            uvs[converted_points[:, 1], converted_points[:, 0]] += 2*uv
        except:
            pass

    return uvs


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    pass


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    pass


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    pass


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    pass


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    pass


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    pass


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    pass


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    pass


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray,
             mask: np.ndarray, levels: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
    pass
