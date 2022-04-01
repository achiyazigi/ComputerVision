"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from ex1_utils import LOAD_GRAY_SCALE, LOAD_RGB, imReadAndConvert
import cv2
import numpy as np


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    title_window = 'Gamma Correction'
    cv2.namedWindow(title_window)
    img = imReadAndConvert(img_path, rep)

    def on_trackbar(gamma):
        # correct image with normalized gamma ([0,200]=>[0,2])
        img2 = np.power(img, gamma/100)
        # convert to BGR [0,255]
        to_show = cv2.cvtColor(
            (img2 * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imshow(title_window, to_show)

    cv2.createTrackbar('Gamma:', title_window, 100, 200, on_trackbar)

    on_trackbar(100)
    cv2.waitKey()
    cv2.destroyAllWindows()


def main():
    gammaDisplay('beach.jpg', LOAD_RGB)


if __name__ == '__main__':
    main()
