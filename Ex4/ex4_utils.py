from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import cv2


def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: Tuple[int, int], k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimum and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    padded_l = cv2.copyMakeBorder(
        img_l, k_size, k_size, k_size, k_size, borderType=cv2.BORDER_CONSTANT, value=0)
    padded_r = cv2.copyMakeBorder(
        img_r, k_size, k_size, k_size, k_size, borderType=cv2.BORDER_CONSTANT, value=0)

    def SSD(I1, I2):
        return np.sum(np.power(I1-I2, 2))

    def correspondence_pixel(y, x):
        y += k_size
        x += k_size
        win_l = padded_l[y-k_size: y+k_size+1, x-k_size: x+k_size+1]
        min_index = k_size
        min_value = np.inf
        for i in range(k_size, img_r.shape[1] - k_size - 1):
            win_r = padded_r[y-k_size: y+k_size+1, i-k_size: i+k_size+1]
            ssd = SSD(win_l, win_r)
            if ssd < min_value:
                min_value = ssd
                min_index = i
        return [y-k_size, min_index-k_size]

    slide = np.array([
        [correspondence_pixel(y, x) for x in range(img_l.shape[1])]
        for y in range(img_l.shape[0])
    ])
    indices = np.dstack(np.indices(img_l.shape))
    disparity = np.abs(indices - slide)
    res = disparity[:, :, 1]
    return res.reshape(img_l.shape)


def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: int, k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: The Maximum disparity range. Ex. 80
    k_size: Kernel size for computing the NormCorolation, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    padded_l = cv2.copyMakeBorder(
        img_l, k_size, k_size, k_size, k_size, borderType=cv2.BORDER_CONSTANT, value=0)
    padded_r = cv2.copyMakeBorder(
        img_r, k_size, k_size, k_size, k_size, borderType=cv2.BORDER_CONSTANT, value=0)

    def cross_corr(I1, I2):
        return np.sum(I1 * I2)/np.sqrt(np.sum(np.power(I1, 2))*np.sum(np.power(I2, 2)))

    def correspondence_pixel(y, x):
        y += k_size
        x += k_size
        win_l = padded_l[y-k_size: y+k_size+1, x-k_size: x+k_size+1]
        max_index = k_size
        max_value = -1
        for i in range(k_size, img_r.shape[1] - k_size - 1):
            win_r = padded_r[y-k_size: y+k_size+1, i-k_size: i+k_size+1]
            cross = cross_corr(win_l, win_r)
            if cross > max_value:
                max_value = cross
                max_index = i
        return [y-k_size, max_index-k_size]
    slide = np.array([
        [correspondence_pixel(y, x) for x in range(img_l.shape[1])]
        for y in range(img_l.shape[0])
    ])
    indices = np.dstack(np.indices(img_l.shape))
    disparity = np.abs(indices - slide)
    res = disparity[:, :, 1]
    return res.reshape(img_l.shape)


def convert_to_hom(src_pnt) -> np.ndarray:
    return np.vstack((src_pnt.T, np.ones(src_pnt.shape[0])))


def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> Tuple[np.ndarray, float]:
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
    fig1 = plt.figure()

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

    ##### Your Code Here ######

    # out = dst_img * mask + src_out * (1 - mask)
    # plt.imshow(out)
    # plt.show()
