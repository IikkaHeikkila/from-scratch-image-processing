import numpy as np
from skimage import io, restoration, filters
import matplotlib.pyplot as plt
from scipy import ndimage


def gradient_inpaint_1(img):
    shape = np.shape(img)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img[i][j] == 0:
                img[i][j] = img[i][j-1]
    return img
    

def gradient_inpaint_2(img):
    x1, y1 = 655, 110
    x2, y2 = 860, 250
    for i in range(y1, y2):
        c1 = img[i][x1-1]
        c2 = img[i][x2+1]
        c0 = img[i][x1-1]
        for j in range(x1, x2):
            img[i][j] = c0
            c0 += (c2-c1)/(x2-x1)
            
    x1, y1 = 155, 860
    x2, y2 = 350, 920
    for i in range(y1, y2):
        c1 = img[i][x1-1]
        c2 = img[i][x2+1]
        c0 = img[i][x1-1]
        for j in range(x1, x2):
            img[i][j] = c0
            c0 += (c2-c1)/(x2-x1)
    
    return img





file = "ouluInpaint .npy"
img = np.load(file).astype(np.float32)


threshold = 0
mask = (img <= threshold).astype(np.float32)


print(img.shape)
print(mask.shape)



inpaint_img = gradient_inpaint_2(img)

img = np.load(file).astype(np.float32)
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Input Image')
plt.subplot(1, 2, 2)
plt.imshow(inpaint_img, cmap='gray')
plt.title('Inpainted Image')
plt.show()


x1, y1 = 650, 110
x2, y2 = 860, 250


















