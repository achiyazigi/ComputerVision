from ex2_main import *
import numpy as np
import cv2

sig = np.array([[3, 6, 1, 7, 2, 6, 0, 4],
                [2, 8, 1, 3, 2, 2, 4, 9],
                [3, 6, 3, 7, 2, 8, 4, 4],
                [3, 6, 3, 2, 2, 7, 4, 1]], dtype=np.float64)
sig_min: np.float64 = sig.min()
sig -= sig_min
sig /= sig.max()
ker = np.array([[2, 3, 3],
                [2, 1, 1],
                [6, 2, 7]], dtype=np.float64)
ker /= ker.sum()

convDerivative(sig)
