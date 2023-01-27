import enum
import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_image(image_1, image_2, title_1="Original", title_2="New Image") :
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(image_1, cmap="gray")
    plt.title(title_1)
    plt.subplot(1,2,2)
    plt.imshow(image_2, cmap="gray")
    plt.title(title_2)
    plt.show()


def plot_hist(old_image, new_image, title_old="Original", title_new="New Image") :
    intensity_values = np.array([x for x in range(256)])
    plt.subplot(1,2,1)
    plt.bar(intensity_values, cv2.calcHist([old_image], [0], None, [256], [0,256])[:,0],width=5)
    plt.title(title_old)
    plt.xlabel('intensity')
    plt.subplot(1,2,2)
    plt.bar(intensity_values, cv2.calcHist([new_image], [0], None, [256], [0,256])[:,0],width=5)
    plt.title(title_new)
    plt.xlabel('intensity')
    plt.show()


toy_image = np.array([[0,2,2],[1,1,1],[1,1,2]], dtype=np.uint8)
# plt.imshow(toy_image, cmap="gray")
# plt.show()
# print("toy_image: ", toy_image)

# plt.bar([x for x in range(6)], [1,5,2,0,0,0])
# plt.show()

# plt.bar([x for x in range(6)], [0,1,0,5,0,2])
# plt.show()

goldhill = cv2.imread("/Users/wng/Desktop/Computer Vision/CV Labs/goldhill.bmp", cv2.IMREAD_GRAYSCALE)
# plt.figure(figsize=(8,8))
# plt.imshow(goldhill, cmap="gray")
# plt.show()

hist = cv2.calcHist([goldhill],[0], None, [256], [0,256])
# plt.plot(hist)
# plt.show()
intensity_values = np.array([x for x in range(hist.shape[0])])
plt.bar(intensity_values, hist[:,0], width=5)
plt.title("Bar histogram")
plt.show()

PMF = hist / (goldhill.shape[0] * goldhill.shape[1])
# plt.plot(PMF)
# plt.show()

# plt.plot(intensity_values, hist)
# plt.title("histogram")
# plt.show()

SB_array = cv2.imread("/Users/wng/Desktop/Computer Vision/CV Labs/baboon.png")
color = ('blue', 'green', 'red')
# for i, col in enumerate(color):
#     histr = cv2.calcHist([SB_array], [i], None, [256], [0,256])
#     plt.plot(intensity_values, histr, color = col, label=col+" channel")
#     plt.xlim([0,256])
# plt.legend()
# plt.title("Histogram Channels")
# plt.show()

neg_toy_image = -1 * toy_image + 255
# print("toy image\n", toy_image)
# print("image negatives\n", neg_toy_image)

# plt.figure(figsize=(8,8))
# plt.subplot(121)
# plt.imshow(toy_image, cmap='gray')
# plt.subplot(122)
# plt.imshow(neg_toy_image,cmap='gray')
# plt.show()
# print('toy_image:', toy_image)

mammogram_array = cv2.imread("/Users/wng/Desktop/Computer Vision/CV Labs/mammogram.png", cv2.IMREAD_GRAYSCALE)
cv2.rectangle(mammogram_array, pt1=(160,212), pt2=(250,289), color=(255), thickness=2)
# plt.figure(figsize=(8,8))
# plt.imshow(mammogram_array, cmap='gray')
# plt.show()

neg_mammogram = -1 * mammogram_array + 255
# plt.figure(figsize=(8,8))
# plt.imshow(neg_mammogram, cmap='gray')
# plt.show()

alpha = 1
beta = 100
new_goldhill = cv2.convertScaleAbs(goldhill, alpha=alpha, beta=beta)
# plot_image(goldhill, new_goldhill, title_1="Original", title_2="Brightness Control")
# plt.figure(figsize=(8,8))
# plot_hist(goldhill, new_goldhill, "Original", "Brightness Control")

alpha = 2
beta = 0
new_goldhill_2 = cv2.convertScaleAbs(goldhill, alpha=alpha, beta=beta)
# plot_image(goldhill, new_goldhill_2, title_1="Original", title_2="Brightness Control")
# plt.figure(figsize=(8,8))
# plot_hist(goldhill, new_goldhill_2, "Original", "Brightness Control")

alpha = 3
beta = -200
new_goldhill_3 = cv2.convertScaleAbs(goldhill, alpha=alpha, beta=beta)
# plot_image(goldhill, new_goldhill_3, title_1="Original", title_2="Brightness Control")
# plt.figure(figsize=(8,8))
# plot_hist(goldhill, new_goldhill_3, "Original", "Brightness Control")

zelda_array = cv2.imread("/Users/wng/Desktop/Computer Vision/CV Labs/zelda.png", cv2.IMREAD_GRAYSCALE)
equalized_zelda = cv2.equalizeHist(zelda_array)
# plot_image(zelda_array, equalized_zelda, 'Original', 'Histogram Equalization')
# plt.figure(figsize=(8,8))
# plot_hist(zelda_array, equalized_zelda, 'Original', 'Histogram Equalization')

bike = cv2.imread("/Users/wng/Desktop/Computer Vision/CV Labs/bike.png", cv2.IMREAD_GRAYSCALE)
# equalized_bike = cv2.equalizeHist(bike)
# plt.imshow(equalized_bike, cmap="gray")
# plt.show()

def thresholding(input_img, threshold, max_value = 255, min_value = 0) :
    N, M = input_img.shape
    output_img = np.zeros((N, M), dtype=np.uint8)

    for i in range(N):
        for j in  range(M):
            if input_img[i,j] > threshold:
                output_img[i,j] = max_value
            else:
                output_img[i,j] = min_value

    return output_img


threshold = 1
max_value = 2
min_value = 0
thresholding_toy = thresholding(toy_image, threshold=threshold, max_value=max_value, min_value=min_value)
# plt.figure(figsize=(8,8))
# plt.subplot(121)
# plt.imshow(toy_image, cmap='gray')
# plt.title('Original Image')
# plt.subplot(122)
# plt.imshow(thresholding_toy, cmap='gray')
# plt.title('Image After Thresholding')
# plt.show()

cameraman_array = cv2.imread('/Users/wng/Desktop/Computer Vision/CV Labs/cameraman.jpeg', cv2.IMREAD_GRAYSCALE)
# plt.figure(figsize=(8,8))
# plt.imshow(cameraman_array, cmap='gray')
# plt.show()


cameraman_hist = cv2.calcHist([cameraman_array], [0], None, [256], [0,256])
# plt.plot(cameraman_hist)
# plt.show()
# plt.bar(intensity_values, cameraman_hist[:,0], width=5)
# plt.title('Bar Histogram : "cameraman.jpeg"')
# plt.show()

threshold = 87
max_value = 255
min_value = 0
thresholding_cameraman = thresholding(cameraman_array, threshold=threshold, max_value=max_value, min_value=min_value)
# plot_image(cameraman_array, thresholding_cameraman, 'Original', 'Image After Thresholding')
# plt.figure(figsize=(8,8))
# plot_hist(cameraman_array, thresholding_cameraman, 'Original', 'Image After Thresholding')

ret, new_cameraman = cv2.threshold(cameraman_array, threshold, max_value, cv2.THRESH_BINARY)
# plot_image(cameraman_array, new_cameraman, 'Original', 'Image After Thresholding : THRESH_BINARY')
# plot_hist(cameraman_array, new_cameraman, 'Original', 'Image After Thresholding : THRESH_BINARY')

ret, new_cameraman = cv2.threshold(cameraman_array, 86, 255, cv2.THRESH_TRUNC)
# plot_image(cameraman_array, new_cameraman, 'Original', 'Image After Thresholding : THRESH_TRUNC')
# plot_hist(cameraman_array, new_cameraman, 'Original', 'Image After Thresholding : THRESH_TRUNC')

ret, otsu = cv2.threshold(cameraman_array, 0, 255, cv2.THRESH_OTSU)
# plot_image(cameraman_array, otsu, 'Original', 'Image After Thresholding : THRESH_OTSU')
# plot_hist(cameraman_array, otsu, 'Original', 'Image After Thresholding : THRESH_OTSU')

