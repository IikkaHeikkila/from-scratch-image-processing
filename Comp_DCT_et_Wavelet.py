import numpy as np
import pywt
from scipy.fftpack import dct, idct
from skimage import io, color
import matplotlib.pyplot as plt


def relative_error(ref, rec):
    diff1 = rec - ref
    sq_diff1 = diff1**2
    sum_sq_diff1 = np.sum(sq_diff1)
    dist1 = np.sqrt(sum_sq_diff1)
    diff2 = ref - np.zeros_like(ref)
    sq_diff2 = diff2**2
    sum_sq_diff2 = np.sum(sq_diff2)
    dist2 = np.sqrt(sum_sq_diff2)
    result = dist1**2 / dist2**2
    return result

def count_zero_coefficients(matrix):
    num_zeros = np.sum(matrix == 0)
    return num_zeros

def threshold_coefficients(matrix, threshold):
    """Set all coefficients below a certain threshold to zero."""
    matrix[np.abs(matrix) < threshold] = 0
    return matrix
    
def dct_coefs(gray_img_norm, block_size=8):
    dct_img = np.zeros_like(gray_img_norm)
    for i in range(0, gray_img_norm.shape[0], block_size):
        for j in range(0, gray_img_norm.shape[1], block_size):
            block = gray_img_norm[i:i+block_size, j:j+block_size]
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            dct_img[i:i+block_size, j:j+block_size] = dct_block
    return dct_img

def dct_rec(dct_img, block_size=8):
    reconstructed_img = np.zeros_like(dct_img)
    for i in range(0, dct_img.shape[0], block_size):
        for j in range(0, gray_img_norm.shape[1], block_size):
            dct_block = dct_img[i:i+block_size, j:j+block_size]
            block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
            reconstructed_img[i:i+block_size, j:j+block_size] = block
    return reconstructed_img


def wavelet_transform(img, wavelet='haar', level=1):
    coeffs = pywt.dwt2(img, 'haar')
    thresh = 0     #
    coeffs = [np.where(np.abs(c) < thresh, 0, c) for c in coeffs]
    ze_cofs = np.sum(coeffs[0] == 0) + np.sum(coeffs[1][0] == 0) + np.sum(coeffs[1][1] == 0) + np.sum(coeffs[1][2] == 0)
    print("---", ze_cofs)
    return coeffs
    
    

def wavelet_reconstruct(coeffs, wavelet='haar'):
    img = pywt.idwt2(coeffs, 'haar')
    return img
    
    

img = io.imread('Test_img_1.jpg') #<--- Your file here
img = np.array(img)

#Monochromatic and normalization
gray_img = color.rgb2gray(img)
gray_img_norm = gray_img.astype(np.float32) / np.max(gray_img)


#DCT and REC with threshold
dct_img = dct_coefs(gray_img_norm)
thr_img = threshold_coefficients(dct_img, threshold=0.20)
print(count_zero_coefficients(dct_img))
reconstructed_img = dct_rec(thr_img)
print(relative_error(gray_img_norm, reconstructed_img))


plt.subplot(121)
plt.imshow(gray_img_norm, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(122)
plt.imshow(reconstructed_img, cmap='gray')
plt.title('Reconstructed Image')
plt.axis('off')
plt.show()







