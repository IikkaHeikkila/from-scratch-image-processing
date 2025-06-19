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
    
    

img = io.imread('Test_img_1.jpg')
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


"""
#Wav and REC with threshold
coeffs = wavelet_transform(gray_img_norm)
reconstructed_img = wavelet_reconstruct(coeffs)
reconstructed_img = (reconstructed_img - np.min(reconstructed_img)) / (np.max(reconstructed_img) - np.min(reconstructed_img))
print(relative_error(gray_img_norm, reconstructed_img))
"""



plt.subplot(121)
plt.imshow(gray_img_norm, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(122)
plt.imshow(reconstructed_img, cmap='gray')
plt.title('Reconstructed Image')
plt.axis('off')
plt.show()




"""
#Image_1
x1 = [48108, 1630694, 1851896, 1985177, 2036211, 2069372, 2089488, 2101343, 2109321, 2114394, 2117871]
y1 = [1.5650302287666684e-14, 0.000279052804816305, 0.002125053010609617, 0.005137181267500088,
      0.007437286043084238, 0.009865066270571434, 0.012080120106224229, 0.013900539432241635,
      0.015530604951591778, 0.016862500744105467, 0.018006048189518294]


x2 = [224264, 1291071, 1460429, 1536542, 1574939, 1598416, 1615527, 1631232, 1647898, 1664895, 1681679]
y2 = [8.93528925455567e-15, 0.0007252866072432973, 0.0025916428148622358, 0.0055414312889165126,
      0.00800003105509178, 0.010381423382459646, 0.012141062219937946, 0.013346878016111285,
      0.014088250681209467, 0.018853812626321412, 0.024763511577078554]

plt.plot(x1, y1, label='DCT')
plt.plot(x2, y2, label='Wavelet (haar)')

plt.xlabel('Num zero-coeffs')
plt.ylabel('Relative error')
plt.title('High-detailed image')
plt.legend()


plt.show()



#Iage_2
x1 = [12236, 271589, 280365, 285370, 288596, 290950, 292687, 294186, 295361, 296329, 297123]
y1 = [1.480297363932996e-14, 5.8249816354283514e-05, 0.0003736502102939336, 0.0008538310456855713,
      0.0014832061696280982, 0.002249643871688703, 0.0030901651985875004, 0.0040889049403465775,
      0.005139963525260463, 0.00625614025701158, 0.007398458854805071]

x2 = [59412, 210861, 217229, 220877, 224107, 226239, 228839, 230982, 232859, 234218, 235410]
y2 = [5.899386658653637e-15, 0.0005546888897045021, 0.002116411220633687, 0.004983491398896155,
      0.009149989614870535, 0.01216223156190756, 0.017495644662397607, 0.004719973639568084,
      0.006379201924135618, 0.007939645400593642, 0.009642020087903342]

plt.plot(x1, y1, label='DCT')
plt.plot(x2, y2, label='Wavelet (haar)')

plt.xlabel('Num zero-coeffs')
plt.ylabel('Relative error')
plt.title('Simple image')
plt.legend()

plt.show()
"""




















