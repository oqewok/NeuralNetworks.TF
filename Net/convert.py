import numpy as np
import matplotlib.pyplot as plt

from skimage import io

image = io.imread('E:/data/gt_db/s01/01.jpg')
a = image[0]/255
plt.imshow(image)
plt.waitforbuttonpress()

print(image.shape)

