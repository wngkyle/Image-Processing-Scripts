from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_concat_h(im1, im2):
    dst = Image.new("RGB", (im1.width + im2.width, im1.height))
    dst.paste(im1, (0,0))
    dst.paste(im2, (im1.width, 0))
    return dst

baboon_image = cv2.imread("baboon.png")
print(type(baboon_image))
# plt.figure(figsize=(5,5))
# plt.imshow(baboon_image)
# plt.show()
# new_baboon_image = cv2.cvtColor(baboon_image, cv2.COLOR_BGR2RGB)
# plt.imshow(new_baboon_image)
# plt.show()
# baboon_image_gray = cv2.cvtColor(baboon_image, cv2.COLOR_BGR2GRAY)
# plt.imshow(baboon_image_gray, cmap='gray')
# plt.show()

# cv2.imwrite("baboon_gray.png", baboon_image_gray)
# new_baboon_image_gray = cv2.imread('baboon_gray.png', cv2.IMREAD_GRAYSCALE)
# plt.imshow(new_baboon_image_gray, cmap = 'gray')
# plt.show()

baboon = cv2.imread("baboon.png")
# plt.figure(figsize=(5,5))
# plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
# plt.show()

blue, green, red = baboon[:,:,0], baboon[:,:,1], baboon[:,:,2]
im_bgr = cv2.vconcat([blue, green, red])
im_rgb = cv2.vconcat([red, green, blue])
# plt.imshow(im_bgr)
# plt.show()
# plt.imshow(im_rgb)
# plt.show()

# plt.figure(figsize=(5,5))
# plt.subplot(121)
# plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
# plt.title('RGB image')
# plt.subplot(122)
# plt.imshow(im_bgr, cmap = 'gray')
# plt.title("Different color channels blue(top), green(middle), red(buttom)")
# plt.show()

# rows = 256
# new_image = cv2.imread("SB.png")
# plt.figure(figsize=(10,10))
# plt.imshow(new_image[0:rows,:,:])
# plt.show()

# columns = 256
# plt.figure(figsize=(10,10))
# plt.imshow(new_image[:,0:256,:])
# plt.show()

# new_image = baboon.copy()
# new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(5,5))
# plt.imshow(new_image[:,0:256,:])
# plt.show()
# plt.imshow(new_image[0:256,:,:])
# plt.show()

# Question 1
barbara = cv2.imread("barbara.png")
barbara = cv2.cvtColor(barbara, cv2.COLOR_BGR2RGB)
barbara[:,:,2] = 0
# plt.figure(figsize=(5,5))
# plt.imshow(barbara)
# plt.show()