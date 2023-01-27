import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def plot_image(image_1, image_2, title_1 = "Original", title_2 = "New Image"):
    plt.figure(figsize=(10,10))
    plt.subplot(121)
    plt.imshow(image_1)
    plt.title(title_1)
    plt.subplot(122)
    plt.imshow(image_2)
    plt.title(title_2)
    plt.show()

image = Image.open("/Users/wng/Desktop/Computer Vision/CV Labs/lenna.png")
# plt.figure(figsize=(5,5))
# plt.imshow(image)
# plt.show()


rows, cols = image.size
noise = np.random.normal(0, 15, (rows, cols, 3)).astype(np.uint8)
# print(noise)
noisy_image = image + noise 
noisy_image = Image.fromarray(noisy_image)
# plot_image(image, noisy_image, title_1="Original", title_2="Image Plus Noise")

from PIL import ImageFilter

### 5 by 5 filter
kernel = np.ones((5,5))/36 
kernel_filter = ImageFilter.Kernel((5,5), kernel.flatten())
# np.flatten returns a copy of the array collapsed into one dimension
image_filtered = noisy_image.filter(kernel_filter)
# plot_image(image_filtered, noisy_image, title_1 = "Filtered Image", title_2 = "Image Plus Noise")

### 3 by 3 filter
kernel = np.ones((3,3))/36 
kernel_filter = ImageFilter.Kernel((3,3), kernel.flatten())
image_filtered = noisy_image.filter(kernel_filter)
# plot_image(image_filtered, noisy_image, title_1 = "Filtered Image", title_2 = "Image Plus Noise")

### Gausian Blur 
image_filtered = noisy_image.filter(ImageFilter.GaussianBlur) # default blur radius is 2
# plot_image(image_filtered, noisy_image, title_1 = "Filtered Image", title_2 = "Image Plus Noise")

### 4 by 4 filter
image_filtered = noisy_image.filter(ImageFilter.GaussianBlur(4))
# plot_image(image_filtered, noisy_image, title_1 = "Filtered Image", title_2 = "Image Plus Noise")

### Image Sharpening
kernel = np.array([[-1,-1,-1], 
                   [-1, 9,-1],
                   [-1,-1,-1]])
kernel = ImageFilter.Kernel((3,3), kernel.flatten())
sharpened = image.filter(kernel)
# plot_image(sharpened , image, title_1="Sharpened image",title_2="Image")

### Using predefined image sharpening kernel
sharpened = image.filter(ImageFilter.SHARPEN)
# plot_image(sharpened , image, title_1="Sharpened image",title_2="Image")


### Edges
img_gray = Image.open('/Users/wng/Desktop/Computer Vision/CV Labs/barbara.png')
# plt.imshow(img_gray, cmap = 'gray')
# plt.show()

### Enhancing the edges using predefined kernel
img_gray = img_gray.filter(ImageFilter.EDGE_ENHANCE)
# plt.imshow(img_gray, cmap='gray')
# plt.show()

### Using FIND_EDGES filter
img_gray = img_gray.filter(ImageFilter.FIND_EDGES)
# plt.imshow(img_gray, cmap='gray')
# plt.show()


### Median 
image = Image.open('/Users/wng/Desktop/Computer Vision/CV Labs/cameraman.jpeg')
# plt.imshow(image, cmap='gray')
# plt.show()

### Blurs the background, increasing the segmentation between the cameraman and the background
image = image.filter(ImageFilter.MedianFilter)
# plt.imshow(image, cmap='gray')
# plt.show()


image = Image.open("/Users/wng/Desktop/Computer Vision/CV Labs/lenna.png")
plt.subplot(121)
plt.imshow(image)
image_filtered = image.filter(ImageFilter.MinFilter())
plt.subplot(122)
plt.imshow(image_filtered)
plt.show()