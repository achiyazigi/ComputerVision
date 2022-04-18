from ex2_main import *
import numpy as np
import cv2

# sig = np.array([[3, 6, 1, 7, 2, 6, 0, 4],
#                 [2, 8, 1, 3, 2, 2, 4, 9],
#                 [3, 6, 3, 7, 2, 8, 4, 4],
#                 [3, 6, 3, 2, 2, 7, 4, 1]], dtype=np.float64)
# sig_min: np.float64 = sig.min()
# sig -= sig_min
# sig /= sig.max()
# ker = np.array([[2, 3, 3],
#                 [2, 1, 1],
#                 [6, 2, 7]], dtype=np.float64)
# ker /= ker.sum()

# convDerivative(sig)


# def conv_with_itteration(sig, kernel, itterations) -> np.ndarray:
#     for i in range(itterations):
#         sig = conv1D(sig, kernel)
#     return sig


# temp_kernel = np.array([1, 1])
# itterations = max(0, 5 - temp_kernel.size)
# coef_kernel = conv_with_itteration(
#     temp_kernel, temp_kernel, itterations).astype(np.float64)
# kernel = np.atleast_2d(coef_kernel).T @ [coef_kernel]
# s = kernel.sum()
# kernel /= s
# print(kernel)
# z = np.zeros((5, 5))
# z[2, 2] = 1
# z = cv2.GaussianBlur(z, z.shape, cv2.BORDER_REPLICATE)
# g = cv2.getGaussianKernel(5, -1)
# g /= g[0, 0]
# print(g @ g.T)
# print(coef_kernel)
k = cv2.getGaussianKernel(5, -1)
k /= k[0, 0]
print(k @ k.T)
kernel = k @ k.T
kernel /= kernel.sum()
print(kernel)
a = np.array([1,1,1])
print(np.atleast_2d(a))