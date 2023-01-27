import matplotlib.pyplot as plt
import cv2
import numpy as np

def plot_image(image_1, image_2,title_1="Orignal",title_2="New Image"):
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB))
    plt.title(title_2)
    plt.show()

### Linear Filtering 
image = cv2.imread("/Users/wng/Desktop/Computer Vision/CV Labs/lenna.png")
# print(image)
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.show()

rows, cols, _ = image.shape
noise = np.random.normal(0, 15, (rows, cols, _)).astype(np.uint8)
noisy_image = image + noise
# plot_image(image, noisy_image, title_1="Original", title_2="Image Plus Noise")

### Filtering Noise (Low pass filter)
kernel = np.ones((6,6))/36
image_filtered = cv2.filter2D(src=noisy_image, ddepth=-1, kernel=kernel)
# plot_image(image_filtered, noisy_image, title_1='Filtered Image', title_2='Image Plus Noise')

### Smaller filter
kernel = np.ones((4,4))/16
image_filtered = cv2.filter2D(src=noisy_image, ddepth=-1, kernel=kernel)
# plot_image(image_filtered, noisy_image, title_1='filtered image', title_2='original')


### Gaussian Blur ((5,5), sigmas = 4)
image_filtered = cv2.GaussianBlur(noisy_image, (5,5), sigmaX = 4, sigmaY = 4)
# plot_image(image_filtered, noisy_image, title_1="Filtered image",title_2="Image Plus Noise")

### ((11,11), sigmas = 10)
image_filtered = cv2.GaussianBlur(noisy_image, (11,11), sigmaX = 10, sigmaY = 10)
# plot_image(image_filtered, noisy_image, title_1="Filtered image",title_2="Image Plus Noise")

### Image Sharpening
kernel = np.array([[-1,-1,-1], 
                   [-1, 9,-1],
                   [-1,-1,-1]])
sharpened = cv2.filter2D(image, -1, kernel)
# plot_image(sharpened , image, title_1="Sharpened image",title_2="Image")

### Edges
img_gray = cv2.imread('/Users/wng/Desktop/Computer Vision/CV Labs/barbara.png', cv2.IMREAD_GRAYSCALE)
# print(img_gray)
# plt.subplot(121)
# plt.imshow(img_gray, cmap='gray')

# GaussianBlur decreases changes that may be caused by noise that would affect the gradient
img_gray = cv2.GaussianBlur(img_gray, (3,3), sigmaX=0.1, sigmaY=0.1)
# plt.subplot(122)
# plt.imshow(img_gray, cmap='gray')
# plt.show()

ddepth = cv2.CV_16S
### grad_x
grad_x = cv2.Sobel(src=img_gray, ddepth=ddepth, dx=1, dy=0, ksize=3)
# plt.imshow(grad_x, cmap='gray')
# plt.show()

### grad_y
grad_y = cv2.Sobel(src=img_gray, ddepth=ddepth, dx=0, dy=1, ksize=3)
# plt.imshow(grad_y, cmap='gray')
# plt.show()

abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)

grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
# plt.imshow(grad, cmap='gray')
# plt.show()

### Median
image = cv2.imread('/Users/wng/Desktop/Computer Vision/CV Labs/cameraman.jpeg', cv2.IMREAD_GRAYSCALE)
# plt.imshow(image, cmap='gray')
# plt.show()

filtered_image = cv2.medianBlur(image, 5)
# plt.imshow(filtered_image, cmap='gray')
# plt.show()

ret, outs = cv2.threshold(src=image, thresh=0, maxval = 255, type = cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
# plt.imshow(outs, cmap='gray')
# plt.show()



