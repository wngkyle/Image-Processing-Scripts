import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

# Helper Function
def plot_image(image_1, image_2,title_1="Orignal",title_2="New Image"):
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.imshow(image_1,cmap="gray")
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(image_2,cmap="gray")
    plt.title(title_2)
    plt.show()


image = Image.open("/Users/wng/Desktop/Computer Vision/CV Labs/lenna.png")
# plt.imshow(image)
# plt.show()

# Horizontal expansion
width, height = image.size
new_width = 2 * width 
new_height = height
new_image = image.resize((new_width, new_height))
# plt.imshow(new_image)
# plt.show()

# Vertical expansion
width, height = image.size
new_width = width 
new_height = height * 2
new_image = image.resize((new_width, new_height))
# plt.imshow(new_image)
# plt.show()

# Double both width and height
width, height = image.size
new_width = width * 2
new_height = height * 2
new_image = image.resize((new_width, new_height))
# plt.imshow(new_image)
# plt.show()

# Shrink the image by 1/2
new_width = width // 2
new_height = height // 2
new_image = image.resize((new_width, new_height))
# plt.imshow(new_image)
# plt.show()

# Rotation
theta = 45
new_image = image.rotate(theta)
# plt.imshow(new_image)
# plt.show()

# Mathematical Operations
image = np.array(image)
new_image = image + 20
# plt.imshow(new_image)
# plt.show()

##### Practice visualizing change in intensity values ########
hist = cv2.calcHist([new_image],[0], None, [256], [0,256])
intensity_values = np.array([x for x in range(hist.shape[0])])
# plt.subplot(121)
# plt.bar(intensity_values, hist[:,0])
# plt.title("New")
hist = cv2.calcHist([image],[0], None, [256], [0,256])
# plt.subplot(122)
# plt.bar(intensity_values, hist[:,0])
# plt.title("Original")
# plt.show()
# print(hist.shape)
############################################################

# Noise Generation
new_image = 10 * image 
# plt.imshow(new_image)
# plt.show()
# print(new_image.shape)

# Noise sample with the same shape as new_image is generated
Noise = np.random.normal(0, 20, (height, width, 3)).astype(np.uint8)
# print(Noise.shape)

new_image = image + Noise
# plt.imshow(new_image)
# plt.show()

new_image = image * Noise 
# plt.imshow(new_image)
# plt.show()


# Matrix Operations
from PIL import ImageOps
im_gray = Image.open("/Users/wng/Desktop/Computer Vision/CV Labs/barbara.png")
im_gray = ImageOps.grayscale(im_gray)
im_gray = np.array(im_gray)
# print(im_gray.shape)
# plt.imshow(im_gray, cmap="gray")
# plt.show()

# Singular Value Decomposition
U, s, V = np.linalg.svd(im_gray, full_matrices = True) # s is not rectangle
S = np.zeros((im_gray.shape[0], im_gray.shape[1]))
S[:image.shape[0], :image.shape[0]] = np.diag(s)
# plot_image(U, V, title_1="Matrix U", title_2="Matrix V")
# plt.imshow(S, cmap="gray")
# plt.show()

B = S.dot(V)
# plt.imshow(B, cmap = "gray")
# plt.show()

A = U.dot(B)
# plt.imshow(A,cmap='gray')
# plt.show()

'''
for n_component in [1,10,100,200, 500]:
    S_new = S[:, :n_component]
    V_new = V[:n_component, :]
    A = U.dot(S_new.dot(V_new))
    plt.imshow(A,cmap='gray')
    plt.title("Number of Components:"+str(n_component))
    plt.show()
'''