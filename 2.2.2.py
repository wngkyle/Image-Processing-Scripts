import matplotlib.pyplot as plt

import numpy as np
import cv2

### Lab ###
bike_array = cv2.imread("bike.png")
bike_array = cv2.cvtColor(bike_array, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(8,8))
# plt.imshow(bike_array)
# plt.show()

width, height, C = bike_array.shape
array_flip = np.zeros((width, height, C), dtype=np.uint8)
for i, row in enumerate(bike_array):
    array_flip[width-1-i,:,:] = row
# plt.figure(figsize=(8,8))
# plt.imshow(array_flip)
# plt.show()


# for flipcode in [0,1,-1]:
#     im_flip = cv2.flip(bike_array, flipcode)
#     plt.imshow(cv2.cvtColor(im_flip, cv2.COLOR_BGR2RGB))
#     plt.title("Flipcode:" + str(flipcode))
#     plt.show()


im_flip = cv2.rotate(bike_array, 0)
# plt.imshow(im_flip)
# plt.show()

flip = {"ROTATE_90_CLOCKWISE" : cv2.ROTATE_90_CLOCKWISE,
        "ROTATE_90_COUNTERCLOCKWISE" : cv2.ROTATE_90_COUNTERCLOCKWISE,
        "ROTATE_180" : cv2.ROTATE_180}

# for key, value in flip.items():
#     im_flip = cv2.rotate(bike_array, value)
#     plt.imshow(im_flip)
#     plt.title(key)
#     plt.show()


startPoint, endPoint = (250, 384), (602,684)
bike_draw = bike_array.copy()
cv2.rectangle(bike_draw, startPoint, endPoint, color=(0,255,0), thickness=3)
# plt.figure(figsize=(8,8))
# plt.imshow(bike_draw)
# plt.show()

bike_draw_2 = bike_array.copy()
bike_draw_2 = cv2.putText(img=bike_draw_2, text="TRIUMPH", org=(10,100), color=(255,255,255), fontFace=4, 
                fontScale=4, thickness=3, lineType=cv2.LINE_8, bottomLeftOrigin=False)
# plt.figure(figsize=(8,8))
# plt.imshow(bike_draw_2)
# plt.show()


# Question 1
im = cv2.imread("SB.png")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im_flip = cv2.flip(im, 0)
im_mirror = cv2.flip(im, 1)
# plt.figure(figsize=(8,8))
# plt.subplot(121)
# plt.imshow(im_flip)
# plt.title("im_flip")
# plt.subplot(122)
# plt.imshow(im_mirror)
# plt.title("im_mirror")
# plt.show()