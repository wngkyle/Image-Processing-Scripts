import os 
my_image = "car.png"
cwd = os.getcwd()
print(cwd)
image_path = os.path.join(cwd, my_image)
print(image_path)


from PIL import Image
image = Image.open("baboon.png")
#image.show()
print(type(image))

import matplotlib.pyplot as plt
# plt.figure(figsize = (5,5))
# plt.imshow(image)
# plt.show()

print("Image size: ", image.size)
print("Image mode: ", image.mode)

from PIL import ImageOps
image_gray = ImageOps.grayscale(image)
print("Image mode for grayscale: ", image_gray.mode)

def get_concat_h(im1, im2):
    dst = Image.new("RGB", (im1.width + im2.width, im1.height))
    dst.paste(im1, (0,0))
    dst.paste(im2, (im1.width, 0))
    return dst

# for n in range(3,8):
#     plt.figure(figsize=(10,10))
#     plt.imshow(get_concat_h(image_gray, image_gray.quantize(256//2**n)))
#     plt.title("256 Quantization Levels left vs {} Quantization Levels right".format(256//2**n))
#     plt.show()

baboon_image = Image.open("baboon.png")
red, green, blue = baboon_image.split()
# get_concat_h(baboon_image, red).show()
# get_concat_h(baboon_image, green).show()
# get_concat_h(baboon_image, blue).show()

import numpy as np
array = np.array(image) # creates a new copy of the image, such that the original one will remain unmodified
array = np.asarray(image) # turns the original image into a numpy array (it's recommended not to manipulate the original image directly)
# print(array)
# print(array.shape)
# print(array.min())
# print(array.max())
# plt.figure(figsize=(5,5))
# plt.imshow(array)
# plt.show()

# Numpy slicing
array_topHalf = array[0:256,:,:]
array_rightHalf = array[:,0:256,:]
# plt.figure(figsize=(5,5))
# plt.imshow(array_topHalf)
# plt.show()
# plt.imshow(array_rightHalf)
# plt.show()

# A = array.copy()
# plt.imshow(A)
# plt.show()

baboon_array = np.array(baboon_image)
# plt.imshow(baboon_array[:,:,0], cmap = 'gray')
# plt.show()

# Practice 1
from PIL import Image
import numpy as np
# blue_lenna = Image.open("lenna.png")
# array_lenna = np.array(blue_lenna)
# blue_array = array_lenna.copy()
# blue_array[:,:,0] = 0
# blue_array[:,:,1] = 0
# plt.figure(figsize=(5,5))
# plt.imshow(blue_array, cmap = "gray")
# plt.show()

''' Answer: 
blue_lenna = Image.open('lenna.png')
blue_array = np.array(blue_lenna)
blue_array[:,:,2] = 0
plt.figure(figsize=(10,10))
plt.imshow(blue_array)
plt.show()
'''
