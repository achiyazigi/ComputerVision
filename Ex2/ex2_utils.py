import math
import numpy as np
import cv2


def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    np.flip(k_size)
    # padding:
    zeros = np.zeros(shape=(k_size.size-1), dtype=in_signal.dtype)
    in_signal = np.concatenate(
        (zeros, in_signal, zeros))

    # tuple dot product:
    # [0 0 3] 4] 1] 5] 0] 0]
    #  ^ ^ ^  ^  ^  ^
    # |4|
    # |3|
    # |1|
    tuples_mult = [np.dot(in_signal[i: i + k_size.size], k_size[::-1])
                   for i in range(in_signal.size-k_size.size+1)]

    return np.array(tuples_mult)


def replicate(src: np.ndarray, vertical: np.uint, horizontal: np.uint) -> np.ndarray:
    """
    My own implementation to border replication.
    """
    if vertical == 0 or horizontal == 0:
        return src
    above = math.ceil(vertical/2)
    below = vertical - above
    left = math.ceil(horizontal/2)
    right = horizontal - left
    rows_above = np.repeat(src[0], above, axis=0)
    res = np.insert(
        src, [0] * above, rows_above, axis=0)
    if below > 0:
        rows_below = np.repeat(res[-1], below, axis=0)
        res = np.insert(
            res, [-1] * below, rows_below, axis=0)
    cols_left = np.transpose([res[:, 0]] * left)

    res = np.insert(
        res, [0] * left, cols_left, axis=1)
    if right > 0:
        cols_right = np.transpose([res[:, -1]] * right)
        res = np.insert(
            res, [-1] * right, cols_right, axis=1)
    return res


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """
    h, w = kernel.shape
    hm = h-1
    wm = w-1
    hm_above = math.ceil(hm/2)
    hm_below = math.floor(hm/2)
    wm_left = math.ceil(wm/2)
    wm_right = math.floor(wm/2)

    # At first i wasn't sure if i can use cv2.copyMakeBorder
    # so i implemented my own replicate function...
    # both ways works:
    # in_image = replicate(in_image, hm, wm)
    in_image = cv2.copyMakeBorder(
        in_image, hm_above, hm_below, wm_left, wm_right, cv2.BORDER_REPLICATE)

    filtered = [[np.sum(in_image[i:i + h, j:j+w] * kernel) for j in range(
        in_image.shape[1] - wm)] for i in range(in_image.shape[0] - hm)]
    return np.array(filtered)


def convDerivative(in_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """
    kernel = np.array([[1, 0, -1]])
    image_grad_x = conv2D(in_image, kernel)
    image_grad_y = conv2D(in_image, kernel.T)
    mag = np.sqrt(np.power(image_grad_x, 2) + np.power(image_grad_y, 2))
    direction = np.tanh(image_grad_y/image_grad_x)
    return direction, mag


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    return


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """

    return


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: opencv solution, my implementation
    """

    return


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges usint "ZeroCrossingLOG" method
    :param img: Input image
    :return: opencv solution, my implementation
    """

    return


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """

    return


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> tuple[
        np.ndarray, np.ndarray]:
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """

    return
