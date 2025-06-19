"""
WORK IN PROGRESS....

TODO:
1. Wavelet
2. Linear diffusion
3. Parona-Malik
"""

from PIL import Image, ImageFilter
import numpy as np
import pywt
from scipy.fftpack import dct, idct
from skimage import io, color
from skimage.restoration import (denoise_wavelet, estimate_sigma)
import matplotlib.pyplot as plt



def add_noise(img, p):
    shape = np.shape(img)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if np.random.uniform(0,1) <= p:
                img[i][j] = 0
    return img


def wavelet_denoise(img):
    img = denoise_wavelet(img, method='BayesShrink', mode='hard')
    return img
    

img = io.imread('Test_img_1.jpg')
img = np.array(img)

#Monochromatic and normalization
img = color.rgb2gray(img)
img_orig = img.astype(np.float32) / np.max(img)

plt.imshow(img_orig, cmap='gray')
plt.show()

img_noisy = add_noise(img_orig, 0.1)

plt.imshow(img_noisy, cmap='gray')
plt.show()

img_denoised = wavelet_denoise(img_noisy)

plt.imshow(img_denoised, cmap='gray')
plt.show()
