import matplotlib.pyplot as plt
import cv2
import numpy as np


def plot_image(image_1, image_2,title_1="Orignal",title_2="New Image"):
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.imshow(image_1,cmap="gray")
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(image_2,cmap="gray")
    plt.title(title_2)
    plt.show()


toy_image = np.zeros((6,6))
toy_image[1:5, 1:5] = 255
toy_image[2:4, 2:4] = 0
# plt.imshow(toy_image, cmap = 'gray')
# plt.show()
# print(toy_image)

new_toy = cv2.resize(toy_image, None, fx = 2, fy = 1, interpolation = cv2.INTER_NEAREST)
# plt.imshow(new_toy, cmap='gray')
# plt.show()

image = cv2.imread("/Users/wng/Desktop/Computer Vision/CV Labs/lenna.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.imshow(image)
# plt.show()

# Horizontal Expansion
new_image = cv2.resize(image, None, fx = 2, fy = 1, interpolation = cv2.INTER_CUBIC)
# plt.imshow(new_image)
# plt.show()
# print("old image shape:", image.shape, "new image shape:", new_image.shape)

# Vertical Expansion
new_image = cv2.resize(image, None, fx=1, fy=2, interpolation=cv2.INTER_CUBIC)
# plt.imshow(new_image)
# plt.show()
# print("old image shape:", image.shape, "new image shape:", new_image.shape)

# Scale horizontal and vertical axis by 2 
new_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
# plt.imshow(new_image)
# plt.show()
# print("old image shape:", image.shape, "new image shape:", new_image.shape)

# Shrink by 1/2
new_image = cv2.resize(image, None, fx=1, fy=0.5, interpolation=cv2.INTER_CUBIC)
# plt.imshow(new_image)
# plt.show()
# print("old image shape:", image.shape, "new image shape:", new_image.shape)

# Specifying the number of rows and columns
rows = 100
cols = 200
new_image = cv2.resize(image, (100, 200), interpolation=cv2.INTER_CUBIC)
# plt.imshow(new_image)
# plt.show()
# print("old image shape:", image.shape, "new image shape:", new_image.shape)

# Translation
tx = 100
ty = 0
M = np.float32([[1,0,tx],[0,1,ty]])
# print(M)

rows, cols, _ = image.shape
new_image = cv2.warpAffine(image, M, (cols, rows))
# plt.imshow(new_image)
# plt.show()

new_image = cv2.warpAffine(image, M, (cols + tx, rows + ty))
# plt.imshow(new_image)
# plt.show()

tx = 0
ty = 50
M = np.float32([[1, 0, tx], [0, 1, ty]])
new_image = cv2.warpAffine(image, M, (cols + tx, rows + ty))
# plt.imshow(new_image)
# plt.show()

# Rotation 
theta = 45.0
M = cv2.getRotationMatrix2D(center=(3,3), angle = theta, scale = 1)
new_toy_image = cv2.warpAffine(toy_image, M, (6,6))
# plot_image(toy_image, new_toy_image, title_1='Original', title_2='Rotated Image')
# print(new_toy_image)

cols, rows, _ = image.shape
M = cv2.getRotationMatrix2D(center = (cols//2-1, rows//2-1), angle = theta, scale = 1)
new_image = cv2.warpAffine(image, M, (cols, rows))
# plt.imshow(new_image)
# plt.show()



# Mathematical Operation (Similar to last lab, almost the same)
new_image = image + 20
# plt.imshow(new_image)
# plt.show()

new_image = 10 * image
# plt.imshow(new_image)
# plt.show()

Noise = np.random.normal(0, 20, (rows, cols, 3)).astype(np.uint8)
# print(Noise.shape)

new_image = image + Noise
# plt.imshow(new_image)
# plt.show()

new_image = image*Noise
# plt.imshow(new_image)
# plt.show()


# Matrix Operation
im_gray = cv2.imread('/Users/wng/Desktop/Computer Vision/CV Labs/barbara.png', cv2.IMREAD_GRAYSCALE)
# print(im_gray.shape)
# plt.imshow(im_gray,cmap='gray')
# plt.show()

U, s, V = np.linalg.svd(im_gray, full_matrices=True)
# print(s.shape) # not a rectangle
# Convert to diagonal matrix
S = np.zeros((im_gray.shape[0], im_gray.shape[1]))
S[:image.shape[0], :image.shape[0]] = np.diag(s)
# plot_image(U,V,title_1="Matrix U ",title_2="matrix  V")
# plt.imshow(S,cmap='gray')
# plt.show()

B = S.dot(V)
# plt.imshow(B, cmap='gray')
# plt.show()

A = U.dot(B)
# plt.imshow(A, cmap='gray')
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
