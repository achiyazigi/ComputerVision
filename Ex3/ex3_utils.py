import math
from re import I
from typing import Callable, List, Tuple

import numpy as np
import cv2
import matplotlib.pyplot as plt


def myID() -> np.uint8:
    """
    Return my ID
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


def move_image(img: np.ndarray, dx, dy) -> np.ndarray:
    t = np.array([[1, 0, dx],
                  [0, 1, dy],
                  [0, 0, 1]])
    res = cv2.warpPerspective(img, t, img.shape[::-1])
    return res


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
    for i in range(k):
        p.append(np.array([img1.copy(), img2.copy()]))
        img1 = cv2.pyrDown(img1, dstsize=(img1.shape[1]//2, img1.shape[0]//2))
        img2 = cv2.pyrDown(img2, dstsize=(img2.shape[1]//2, img2.shape[0]//2))
        if(img1.ndim < 2):
            k = i
            break
    for level in range(k-1, -1, -1):
        pyr1, pyr2 = p[level]
        dx_median, dy_median = np.ma.median(np.ma.masked_where(
            uvs == np.zeros((2)), uvs), axis=(0, 1)).filled(0)
        pyr1 = move_image(pyr1, dx_median, dy_median)
        pts, uv = opticalFlow(pyr1, pyr2, max(
            int(stepSize * math.pow(2, -level)), 1), winSize)
        if pts.size == 0:
            continue
        converted_points = pts * np.power(2, level)
        uvs[converted_points[:, 1], converted_points[:, 0]] += 2*uv

    return uvs


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------

def gradient_descent(im1: np.ndarray, im2: np.ndarray, grad_func: Callable[..., np.ndarray], *args, **kwargs):

    FRACTION = 1
    EPOCHS = 70
    # 1. assume no movement
    t = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]], dtype=np.float32)

    def grad_step_twards(u, v, t):
        t += np.array([[0, 0, u],
                       [0, 0, v],
                       [0, 0, 0]], dtype=np.float32)

    last_e = np.inf
    for epoch in range(EPOCHS):
        # 2. move im1 by the updated movement
        moving_img = cv2.warpPerspective(im1, t, im1.shape[::-1])

    # 3. calculate error
        e = np.mean(np.power(moving_img - im2, 2))
        # to print the error
        # if epoch % 10 == 0:
        #     print(f'{epoch:3}. {e}')
        if (last_e - e) ** 2 < 0.00000000001 and last_e < 0.00000001:
            break
        last_e = e
    # 4. calculate error gradient
        uvs = grad_func(moving_img, im2, *args, **kwargs)
        dx, dy = FRACTION * \
            np.ma.mean(np.ma.masked_where(
                uvs == np.zeros((2)), uvs), axis=(0, 1)).filled(0)

    # 5. add fraction of the gradient to the movement
        grad_step_twards(dx, dy, t)

    # 6. repeat from step 2 until converge

    # 7. return movement
    return t


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    return gradient_descent(im1, im2, opticalFlowPyrLK, int(
        math.log2(np.min(im1.shape))), 20, 15)


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    return gradient_descent_rigid(im1, im2, opticalFlowPyrLK, int(math.log2(min(im1.shape))), 20, 15)


def gradient_descent_rigid(im1: np.ndarray, im2: np.ndarray, grad_func: Callable[..., np.ndarray], *args, **kwargs) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    EPOCHS = 300
    t = np.array([[1, 0, 0],
                  [0, 1, 0]], dtype=np.float32)
    error_func = [np.mean(np.power(im1 - im2, 2))]
    for epoch in range(EPOCHS):
        moving_image = cv2.warpAffine(im1, t, im1.shape[::-1])

        e = np.mean(np.power(moving_image - im2, 2))
        # here you can print the error
        # if epoch % 10 == 0:
        #     print(f'{epoch:3}. {e}')
        if (error_func[-1] - e) ** 2 < 0.001 and e < 0.1:
            break
        error_func.append(e)

        uvs = grad_func(moving_image.astype(np.float32), im2.astype(
            np.float32), *args, **kwargs)
        pts = np.argwhere((uvs[:, :, 0] != 0) | (uvs[:, :, 1] != 0))
        uvs = uvs[pts[:, 0], pts[:, 1]]
        pts = pts[:, 1::-1]
        uvs += pts
        A = np.concatenate((np.repeat(pts, 2, axis=0), np.ones(
            (pts.shape[0]*2, 1)), np.zeros((pts.shape[0]*2, 3))), axis=1)
        A[1::2] = np.concatenate((A[1::2, 3:], A[1::2, :3]), axis=1)
        b = np.array([uvs.flatten()]).T
        grad = np.linalg.lstsq(A, b, rcond=None)[0]
        grad = grad.reshape((2, 3))
        full_t = np.vstack([t, [0, 0, 1]])
        full_grad = np.vstack([grad, [0, 0, 1]])
        t = (full_t @ full_grad)[:2]
    # to display the error function:
    # plt.plot(range(len(error_func)), error_func)
    # plt.show()
    return t


def opticalFlowCrossCorr(im1: np.ndarray, im2: np.ndarray, step_size, win_size):
    half = math.floor(win_size/2)
    uvs = np.zeros((*im1.shape, 2))
    im1 = cv2.copyMakeBorder(im1, half, half, half,
                             half, borderType=cv2.BORDER_CONSTANT, value=0)
    im2 = cv2.copyMakeBorder(im2, half, half, half,
                             half, borderType=cv2.BORDER_CONSTANT, value=0)

    def argcorrelation(win: np.ndarray):
        r = cv2.matchTemplate(im2, win, cv2.TM_CCORR_NORMED)
        # max_corr = -1
        # top_correlation = None
        # win1 = win.copy().flatten() - win.mean()
        # norm1 = np.linalg.norm(win1, 2)
        # for y in range(half, im2.shape[0] - half - 1):
        #     for x in range(half, im2.shape[1] - half - 1):
        #         win2 = im2[y - half: y+half + 1, x - half: x+half+1]

        #         win2 = win2.copy().flatten() - win2.mean()
        #         norms = norm1*np.linalg.norm(win2, 2)

        #         corr = 0 if norms == 0 else np.sum(win1 * win2)/norms
        #         if corr > max_corr:
        #             max_corr = corr
        #             top_correlation = (y, x)
        _minVal, _maxVal, minLoc, maxLoc = cv2.minMaxLoc(r, None)
        return np.flip(maxLoc + np.array([half, half]))

    for y in range(half, im1.shape[0] - half - 1, step_size):
        for x in range(half, im1.shape[1] - half - 1, step_size):
            window = im1[y - half: y+half + 1, x - half: x+half+1]
            top_correlation = argcorrelation(window)
            uvs[y-half, x -
                half] = np.flip(top_correlation - np.array([y, x]))
    return uvs.astype(np.float32)


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    uvs = opticalFlowCrossCorr(im1, im2, 15, 9)
    u, v = np.ma.median(np.ma.masked_where(
        uvs == np.zeros((2)), uvs), axis=(0, 1)).filled(0)
    return np.array([[1, 0, u],
                     [0, 1, v],
                     [0, 0, 1]])


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    return gradient_descent_rigid(im1, im2, opticalFlowCrossCorr, 15, 15)


def interpolate2d(indices: np.ndarray, pallete: np.ndarray) -> np.int32:
    cc = np.ceil(indices).astype(np.int32)
    ff = np.floor(indices).astype(np.int32)
    if np.any(cc >= np.array(pallete.shape[:2])):
        return pallete[ff[0], ff[1]]

    def inter_color(a: np.ndarray, b: np.ndarray, c: np.ndarray):
        """
        interpolate only 1 color
        """
        mult = a @ b @ c
        return (mult//np.prod(cc-ff))[0, 0]

    a = np.array([[cc[0] - indices[0], indices[0] - ff[0]]])
    b = pallete[ff[0]:cc[0]+1, ff[1]:cc[1]+1]
    c = np.array([[cc[1] - indices[1]],
                  [indices[1] - ff[1]]])
    if pallete.ndim < 3:
        return inter_color(a, b, c)
    res = []
    for i in range(b.shape[-1]):
        res.append(inter_color(a, b[:, :, i], c))
    return res


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1
    :param im2: input image 2
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    im2_size_2d = np.prod(im2.shape[:2])

    res = np.zeros_like(im1)
    indices = np.flip(np.indices(im2.shape[:2]).reshape(
        (2, im2_size_2d)), axis=0)
    hom_indices = np.vstack((indices, np.ones(im2_size_2d)))
    mapped_indices = (T @ hom_indices)[:2]

    mapped_indices = np.flip(mapped_indices, axis=0)
    z = np.zeros(2)

    for y in range(res.shape[0]):
        for x in range(res.shape[1]):
            if x + y*res.shape[1] < mapped_indices.shape[1]:
                i = mapped_indices[:, x + y*res.shape[1]].flatten()
                if np.all((z < i) & (i < res.shape[:2])):
                    res[y, x] = interpolate2d(i, im2)

    f, ax = plt.subplots(1, 3)
    ax[0].imshow(im1)
    ax[0].set_title('original')
    ax[1].imshow(res)
    ax[1].set_title('restored')
    ax[2].imshow(im2)
    ax[2].set_title('translated')
    plt.show()
    return res

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
    if levels < 1:
        return []
    res = [img]
    for i in range(levels-1):
        ker = cv2.getGaussianKernel(5, -1)
        res.append(cv2.filter2D(
            res[-1], ddepth=-1, kernel=ker)[::2, ::2])
    return res


def expand_img(img, factor) -> np.ndarray:
    expand = np.repeat(np.repeat(img, 2, axis=0), 2, axis=1)
    expand[::2, 1::factor] = 0
    expand[1::2] = 0
    ker = cv2.getGaussianKernel(5, -1)*4
    expand = cv2.sepFilter2D(expand, -1, ker, ker)/4
    return expand


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    pyr = gaussianPyr(img, levels)
    if levels < 2:
        return pyr
    res = [pyr[-1]]
    for i in range(levels-2, -1, -1):
        orig = pyr[i+1]
        expand = expand_img(orig, 2)
        if expand.shape[0] > pyr[i].shape[0]:
            expand = expand[:-1]
        if expand.shape[1] > pyr[i].shape[1]:
            expand = expand[:, :-1]
        res.insert(0, pyr[i] - expand)

    return res


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    if len(lap_pyr) == 0:
        return np.ndarray()
    if len(lap_pyr) == 1:
        return lap_pyr[0]
    res = lap_pyr[-1]
    for i in range(len(lap_pyr)-2, -1, -1):
        img = lap_pyr[i]
        res = expand_img(res, 2)
        if res.shape[0] > img.shape[0]:
            res = res[:-1]
        if res.shape[1] > img.shape[1]:
            res = res[:, :-1]
        res += img
    return res


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
    l1 = np.array(laplaceianReduce(img_1, levels))
    l2 = np.array(laplaceianReduce(img_2, levels))
    gm = np.array(gaussianPyr(mask, levels))
    l3 = gm*l1+(1-gm)*l2
    return img_1*mask + (1-mask)*img_2, laplaceianExpand(l3)
