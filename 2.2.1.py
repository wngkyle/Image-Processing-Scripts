from functools import update_wrapper
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2



bike = cv2.imread("bike.png")
# plt.figure(figsize=(5,5))
# plt.imshow(cv2.cvtColor(bike, cv2.COLOR_BGR2RGB))
# plt.show()

bike = Image.open("bike.png")
# bike.show()
# plt.figure(figsize=(5,5))
# plt.imshow(bike)
# plt.show()

bike_array = np.array(Image.open("bike.png"))
# plt.figure(figsize=(5,5))
# plt.imshow(bike_array)
# plt.show()

bike_array_2 = bike_array.copy()
# plt.figure(figsize=(5,5))
# plt.imshow(bike_array_2)
# plt.show()

red, green, blue = bike_array[:,:,0], bike_array[:,:,1], bike_array[:,:,2]
# plt.imshow(red, cmap = "gray")
# plt.show()
# plt.imshow(green, cmap = "gray")
# plt.show()
# plt.imshow(blue, cmap = "gray")
# plt.show()

### Lab ###

bike_array = np.array(Image.open("bike.png"))
width, height, c = bike_array.shape
array_flip = np.zeros((width, height, c), dtype=np.uint8)
for i, row in enumerate(bike_array):
    array_flip[width - 1 - i, :, :] = row
# plt.imshow(array_flip)
# plt.show()

from PIL import ImageOps

bike_image = Image.open("bike.png")
bike_flipped = ImageOps.flip(bike_image)
# plt.imshow(bike_flipped)
# plt.show()
bike_mirrored = ImageOps.mirror(bike_image)
# plt.imshow(bike_mirrored)
# plt.show()
bike_transposed = bike_image.transpose(6)
# plt.imshow(bike_transposed)
# plt.show()


flip = {"Flip_LEFT_RIGHT" : Image.FLIP_LEFT_RIGHT, 
        "FLIP_TOP_BOTTOM" : Image.FLIP_TOP_BOTTOM, 
        "ROTATE_90" : Image.ROTATE_90, 
        "ROTATE_180" : Image.ROTATE_180, 
        "ROTATE_270" : Image.ROTATE_270,
        "TRANSPOSE" : Image.TRANSPOSE,
        "TRANSVERSE" : Image.TRANSVERSE}

'''
for key, values in flip.items():
    plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    plt.imshow(bike_image)
    plt.title("Original")
    plt.subplot(1,2,2)
    plt.imshow(bike_image.transpose(values))
    plt.title(key)
    plt.show()
'''

bike_crop = bike_array[305:577,600:996,:]
# plt.figure(figsize=(8,8))
# plt.imshow(bike_crop)
# plt.title("Engine")
# plt.show()
bike_crop_2 = bike_image.crop((600, 305, 996, 577))
# plt.figure(figsize=(8,8))
# plt.imshow(bike_crop_2)
# plt.title("Engine_2")
# plt.show()

bike_change_pixel = bike_crop.copy()
bike_change_pixel[:,:,1:2] = 0
# plt.figure(figsize=(8,8))
# plt.subplot(1,2,1)
# plt.imshow(bike_crop)
# plt.title("Original")
# plt.subplot(1,2,2)
# plt.imshow(bike_change_pixel)
# plt.title("Altered Image")
# plt.show()

from PIL import ImageDraw
bike_draw = bike_image.copy()
bike_fn = ImageDraw.Draw(im=bike_draw)
gas_tank = [779,170,1067,280]
wheel = [1277,393,1152,601,1374,607]
bike_fn.rectangle(xy=gas_tank, fill="blue")
# plt.imshow(bike_draw)
# plt.show()

from PIL import ImageFont
font_path = "/Library/Fonts/TI-Nspire.ttf"
font1 = ImageFont.truetype(font_path, 90)
bike_fn.text(xy=(0,0), text="Triumph", fill=(255,255,255), font=font1)
# plt.imshow(bike_draw)
# plt.show()

upper, lower, left, right = 100, 200, 400, 500
SB_image = Image.open("SB.png")
SB_array = np.array(SB_image)
SB_array[upper:lower,left:right,:] = bike_array[upper:lower,left:right,:]
# plt.figure(figsize=(8,8))
# plt.imshow(SB_array)
# plt.show()

crop_image = bike_image.crop((upper,lower,left,right))
SB_image2 = SB_image.copy()
SB_image2.paste(crop_image, (10,10))
# plt.imshow(SB_image2)
# plt.show()


# Question 1
from PIL import Image, ImageOps, ImageDraw
im = Image.open("SB.png")
im_flip = ImageOps.flip(im)
im_mirror = ImageOps.mirror(im)
# plt.figure(figsize=(8,8))
# plt.subplot(1,2,1)
# plt.imshow(im_flip)
# plt.title("Flipped")
# plt.subplot(1,2,2)
# plt.imshow(im_mirror)
# plt.title("Mirrored")
# plt.show()
