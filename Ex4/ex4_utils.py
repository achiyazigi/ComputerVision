import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import uniform_filter


def _disparity(img_l: np.ndarray, img_r: np.ndarray, disp_range, value_method, collect_method) -> np.ndarray:
    min_depth, max_depth = disp_range
    disparity_resolt = np.zeros(
        (*img_l.shape, max_depth-min_depth))  # ndim = 3
    # for each depth
    for offset in range(min_depth, max_depth):
        # move image
        moved_l = np.roll(img_l, -offset)
        # compute value method
        disparity_resolt[:, :, offset -
                         min_depth] = value_method(moved_l, img_r)
    # collect and return
    return collect_method(disparity_resolt, axis=2)+min_depth


def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range, k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimum and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    def value_method(l, r):
        # Simple sum of squered difference (SSD)
        return uniform_filter(np.square(l - r), k_size)
    return _disparity(img_l, img_r, disp_range, value_method, np.argmin)


def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: int, k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: The Maximum disparity range. Ex. 80
    k_size: Kernel size for computing the NormCorolation, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    norm_l = img_l - uniform_filter(img_l, k_size)
    norm_r = img_r - uniform_filter(img_r, k_size)

    def value_method(l, r):
        # Normalized Cross Correlation
        cross_top = uniform_filter(l * r, k_size)
        sigma_l = uniform_filter(np.square(l), k_size)
        sigma_r = uniform_filter(np.square(r), k_size)
        cross_bottom = np.sqrt(sigma_l * sigma_r)
        return cross_top/cross_bottom
    return _disparity(norm_l, norm_r, disp_range, value_method, np.argmax)


def convert_to_hom(src_pnt) -> np.ndarray:
    return np.vstack((src_pnt.T, np.ones(src_pnt.shape[0])))


def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray):
    """
    Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
    returns the homography and the error between the transformed points to their
    destination (matched) points. Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))

    src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
    dst_pnt: 4+ keypoints locations (x,y) on the destenation image. Shape:[4+,2]

    return: (Homography matrix shape:[3,3], Homography error)
    """
    A = np.concatenate((np.repeat(src_pnt, 2, axis=0), np.ones(
        (src_pnt.shape[0]*2, 1)), np.zeros((src_pnt.shape[0]*2, 3))), axis=1)
    A[1::2] = np.concatenate((A[1::2, 3:], A[1::2, :3]), axis=1)
    m_xx_yy = -src_pnt * dst_pnt
    m_xx = m_xx_yy[:, 0].reshape((-1, 1))
    m_yy = m_xx_yy[:, 1].reshape((-1, 1))
    m_xy = -np.flip(dst_pnt, axis=1) * src_pnt
    add_from_right_1 = np.repeat(m_xx, 2, axis=0)
    add_from_right_1[1::2] = m_xy[:, 0].reshape((-1, 1))
    add_from_right_2 = np.repeat(m_xy[:, 1].reshape((-1, 1)), 2, axis=0)
    add_from_right_2[1::2] = m_yy
    add_from_right_3 = np.array([(-dst_pnt).flatten()]).T
    A = np.hstack((A, add_from_right_1, add_from_right_2, add_from_right_3))

    u, s, vh = np.linalg.svd(A)
    M = vh[-1].reshape(3, 3)/vh[-1, -1]
    src_hom = convert_to_hom(src_pnt)
    dst_hom = convert_to_hom(dst_pnt)
    transtated = M.dot(src_hom)
    error = np.sqrt(np.sum((transtated/transtated[-1]-dst_hom)**2))
    return M, error


def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
    Displays both images, and lets the user mark 4 or more points on each image.
    Then calculates the homography and transforms the source image on to the destination image.
    Then transforms the source image onto the destination image and displays the result.

    src_img: The image that will be 'pasted' onto the destination image.
    dst_img: The image that the source image will be 'pasted' on.

    output: None.
    """

    dst_p = []
    src_p = []
    fig1 = plt.figure()

    def onclick_2(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        src_p.append([x, y])

        if len(src_p) == 4:
            plt.close()
        plt.show()

    def onclick_1(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        dst_p.append([x, y])

        if len(dst_p) == 4:
            plt.close()
        plt.show()

    # display image 1
    cid = fig1.canvas.mpl_connect('button_press_event', onclick_1)
    plt.imshow(dst_img)
    plt.show()
    dst_p = np.array(dst_p)
    cid = fig1.canvas.mpl_disconnect(cid)
    fig1 = plt.figure()
    cid = fig1.canvas.mpl_connect('button_press_event', onclick_2)
    plt.imshow(src_img)
    plt.show()
    src_p = np.array(src_p)

    ##### Your Code Here ######
    h, w = src_img.shape[:2]
    hom, err = computeHomography(src_p, dst_p)
    src_out = cv2.warpPerspective(
        src_img, hom, dst_img.shape[1::-1])
    mask = src_out == 0
    out = dst_img * mask + src_out * (1 - mask)
    plt.imshow(out)
    plt.show()
