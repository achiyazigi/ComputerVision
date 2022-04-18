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
    return np.array(filtered).astype(np.float64)


def convDerivative(in_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """
    kernel = np.array([[1, 0, -1]])
    image_grad_x = conv2D(in_image, kernel)
    image_grad_y = conv2D(in_image, kernel.T)
    mag = np.sqrt(np.power(image_grad_x, 2) +
                  np.power(image_grad_y, 2)).astype(np.float64)
    direction = np.arctan2(image_grad_y, image_grad_x).astype(np.float64)
    return direction, mag


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    def conv_with_itteration(sig, kernel, itterations) -> np.ndarray:
        for _ in range(itterations):
            sig = conv1D(sig, kernel)
        return sig.astype(np.float128)
    temp_kernel = np.array([1, 1])
    itterations = max(0, k_size - temp_kernel.size)
    coef_kernel = conv_with_itteration(temp_kernel, temp_kernel, itterations)
    coef_kernel /= coef_kernel.sum()
    coef_kernel = np.atleast_2d(coef_kernel)
    print(coef_kernel.T)

    kernel = coef_kernel.T @ coef_kernel

    return conv2D(in_image, kernel)


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    kernel = cv2.getGaussianKernel(k_size, -1)
    print(kernel)
    print()
    kernel = kernel @ kernel.T

    return cv2.filter2D(in_image, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    # return cv2.GaussianBlur(in_image, (k_size, k_size), cv2.BORDER_REPLICATE)


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

    img = blurImage2(img, 11)
    lap_ker = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]])
    img = conv2D(img, lap_ker)
    minus = img < 0
    plus = img >= 0
    edges = np.zeros_like(img)
    for i in range(1, img.shape[0]):
        for j in range(1, img.shape[1]):
            if minus[i, j-1] and plus[i, j]:
                edges[i, j] = 1
            elif plus[i, j-1] and minus[i, j]:
                edges[i, j] = 1
            if (minus[i - 1, j] and plus[i, j]):
                edges[i, j] = 1
            elif (plus[i - 1, j] and minus[i, j]):
                edges[i, j] = 1

    return edges


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
    SIZE_THRESH_RATIO = 8000
    edges = cv2.Canny((img*255).astype(np.uint8), 250, 500) / 255
    edges_points_arrays = np.where(edges == 1)
    edges_points = list(zip(*edges_points_arrays))
    votes = {}
    # for every point
    for y, x in edges_points:
        for r in range(min_radius, max_radius):
            # for every theta
            for theta in range(181):
                a = int(x + r * np.cos(np.deg2rad(theta)))
                b = int(y + r * np.sin(np.deg2rad(theta)))
                if a < img.shape[0] and b < img.shape[1]:
                    if (a, b, r) in votes:
                        votes[(a, b, r)] += 1
                    else:
                        votes[(a, b, r)] = 1
    threshold = img.size / \
        SIZE_THRESH_RATIO if img.size > SIZE_THRESH_RATIO else max(
            votes.values()) - 1
    best = list(filter(lambda k: votes[k] > threshold, list(votes.keys())))
    return best


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
