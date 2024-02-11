import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread
from skimage.segmentation import slic,mark_boundaries
from skimage.filters import  gabor_kernel
from scipy import ndimage as ndi


def read_rgb_image_opencv():
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_matrix = np.array(rgb_image)
    return rgb_matrix

def createPixelFeature():
    image_matrix = read_rgb_image_opencv()
    y, x, _ = image_matrix.shape

    coordinates = np.column_stack((np.repeat(np.arange(y), x), np.tile(np.arange(x), y)))
    image_matrix_with_coordinates = np.column_stack((image_matrix.reshape(-1, 3), coordinates))

    image_matrix_with_coordinates = image_matrix_with_coordinates.reshape(y, x, -1)

    normalized_array = image_matrix_with_coordinates.copy().astype(np.float64)

    normalized_array[:, :, :3] /= 255.0
    normalized_array[:, :, 3] /= float(y-1)
    normalized_array[:, :, 4] /= float(x-1)
    return normalized_array

def k_means(matrix, k):
    try:
        h, w, y = matrix.shape
    except ValueError:
        h, y = matrix.shape
        w = 1

    flattened_image = matrix.reshape((h * w, y))
    centroids = flattened_image[np.random.choice(h * w, k, replace=False)]

    for _ in range(100):
        distances = np.linalg.norm(flattened_image[:, np.newaxis, :] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([flattened_image[labels == j].mean(axis=0) for j in range(k)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return centroids,labels

def feature1ToPlot(k):
    feature1 = read_rgb_image_opencv()
    h, w, y = feature1.shape
    centroid1, label1 = k_means(feature1 / 255, k)
    segmented_image = centroid1[label1].reshape((h, w, y))
    labels1 = (segmented_image * 255).astype(np.uint8)

    lastExt = np.copy(segmented_image)
    segmented = np.zeros(np.shape(lastExt)[:2])
    for i in range(len(lastExt)):
        for j in range(len(lastExt[i])):
            for k in range(len(centroid1)):
                if all(x == y for x, y in zip(lastExt[i][j], centroid1[k])):
                    segmented[i][j] = k
                    break

    segmented = segmented.astype(np.uint8)
    original = imread(image_path)
    output1 = mark_boundaries(original, segmented, color=(1, 1, 1))

    return labels1,output1

def feature2ToPlot(k):
    feature2 = createPixelFeature()
    h, w, y = feature2.shape
    centroid2, label2 = k_means(feature2, k)
    segmented_image = centroid2[label2].reshape((h, w, y))
    labels2 = (segmented_image * 255).astype(np.uint8)[:, :, :3]

    lastExt = np.copy(segmented_image[:,:, :3])

    segmented = np.zeros(np.shape(lastExt)[:2])
    for i in range(len(lastExt)):
        for j in range(len(lastExt[i])):
            for k in range(len(centroid2)):
                if all(x == y for x, y in zip(lastExt[i][j], centroid2[k])):
                    segmented[i][j] = k
                    break

    segmented = segmented.astype(np.uint8)
    original = imread(image_path)
    output2 = mark_boundaries(original, segmented, color=(1, 1, 1))


    return labels2,output2

def feature3ToPlot(segment,k):
    img = imread(image_path)
    segments = slic(img, segment)

    superpixelImage = mark_boundaries(img,segments, color=(1,1,1))

    superpixelMatrix = list()

    for i in range(np.max(segments)):
        superpixelMatrix.append([])

    for i in range(len(segments)):
        for j in range(len(segments[i])):
            superpixelMatrix[segments[i][j] - 1].append(img[i][j])

    superPixelMeans = list()

    for i in range(len(superpixelMatrix)):
        r_sum = 0
        g_sum = 0
        b_sum = 0
        for j in range(len(superpixelMatrix[i])):
            r_sum += superpixelMatrix[i][j][0]
            g_sum += superpixelMatrix[i][j][1]
            b_sum += superpixelMatrix[i][j][2]
        r_sum /= len(superpixelMatrix[i])
        g_sum /= len(superpixelMatrix[i])
        b_sum /= len(superpixelMatrix[i])
        superPixelMeans.append([r_sum, g_sum, b_sum])

    centroid3, label3 = k_means(np.array(superPixelMeans), k)
    indexed = centroid3[label3]


    last = indexed[segments - 1] / 255

    lastExt = indexed[segments - 1]
    segmented = np.zeros(np.shape(lastExt)[:2])


    for i in range(len(lastExt)):
        for j in range(len(lastExt[i])):
            for k in range(len(centroid3)):
                if all(x == y for x, y in zip(lastExt[i][j], centroid3[k])):
                    segmented[i][j] = k
                    break

    segmented = segmented.astype(np.uint8)
    output3 = mark_boundaries(img, segmented, color=(1, 1, 1))

    return last, superpixelImage , output3


def feature4ToPlot(segment,k):
    img = imread(image_path)
    segments = slic(img, segment)

    superpixelMatrix = list()

    for i in range(np.max(segments)):
        superpixelMatrix.append([])

    for i in range(len(segments)):
        for j in range(len(segments[i])):
            superpixelMatrix[segments[i][j] - 1].append(img[i][j])


    superpixelHistogramMatrix = list()

    for i in range(len(superpixelMatrix)):
        red_flatten = []
        green_flatten = []
        blue_flatten = []
        for j in range(len(superpixelMatrix[i])):
            pixel = superpixelMatrix[i][j]
            red_flatten.append(pixel[0])
            green_flatten.append(pixel[1])
            blue_flatten.append(pixel[2])

        red_hist, _ = np.histogram(red_flatten, bins=256, range=[0, 256])
        green_hist, _ = np.histogram(green_flatten, bins=256, range=[0, 256])
        blue_hist, _ = np.histogram(blue_flatten, bins=256, range=[0, 256])

        superpixelHistogramMatrix.append(red_hist + green_hist + blue_hist)

    _, label4 = k_means(np.array(superpixelHistogramMatrix), k)

    temp = list()
    for i in range(0, 255, int(255 / k)):
        temp.append([i, i, i])
    temp = np.array(temp)

    indexed = temp[label4]
    last = indexed[segments - 1]

    lastExt = indexed[segments - 1]
    segmented = np.zeros(np.shape(lastExt)[:2])

    for i in range(len(lastExt)):
        for j in range(len(lastExt[i])):
            for k in range(len(temp)):
                if all(x == y for x, y in zip(lastExt[i][j], temp[k])):
                    segmented[i][j] = k
                    break

    segmented = segmented.astype(np.uint8)
    output4 = mark_boundaries(img, segmented, color=(1, 1, 1))


    return last,output4


def feature5ToPlot(segment,k):
    filterbank = []
    frequencies = [0.5,1.5]
    thetas = [0,90,180]

    for freq in frequencies:
        for theta in thetas:
            kernel = np.real(gabor_kernel(frequency=freq, theta=np.deg2rad(theta)))
            filterbank.append(kernel)

    filtered = list()

    for kernel in filterbank:
        img = imread(image_path)
        for i in range(3):
            img[:, :, i] = ndi.convolve(img[:, :, i], kernel, mode='wrap')
        filtered.append(img)

    original_image = imread(image_path)
    meanImage = np.zeros(np.shape(original_image))

    for image in filtered:
        meanImage += image

    meanImage /= len(filtered)

    meanImage = np.array(meanImage).astype(np.uint8)

    segments = slic(img, segment)


    superpixelMatrix = list()
    for i in range(np.max(segments)):
        superpixelMatrix.append([])

    for i in range(len(segments)):
        for j in range(len(segments[i])):
            index = segments[i][j]
            point = meanImage[i][j]
            superpixelMatrix[index - 1].append(point)

    superpixelMean = list()

    for i in range(len(superpixelMatrix)):
        r_sum = 0
        g_sum = 0
        b_sum = 0
        for j in range(len(superpixelMatrix[i])):
            r_sum += superpixelMatrix[i][j][0]
            g_sum += superpixelMatrix[i][j][1]
            b_sum += superpixelMatrix[i][j][2]

        r_sum /= len(superpixelMatrix[i])
        g_sum /= len(superpixelMatrix[i])
        b_sum /= len(superpixelMatrix[i])

        superpixelMean.append([r_sum, g_sum, b_sum])

    centroid5, label5 = k_means(np.array(superpixelMean), k)
    indexed = centroid5[label5]

    for i in range(len(segments)):
        for j in range(len(segments[i])):
            index = segments[i][j] - 1
            meanImage[i][j] = indexed[index]

    lastExt = np.copy(meanImage)

    segmented = np.zeros(np.shape(lastExt)[:2])

    for i in range(len(lastExt)):
        for j in range(len(lastExt[i])):
            for k in range(len(centroid5)):
                if all(x == y for x, y in zip(lastExt[i][j], centroid5[k].astype(np.uint8))):
                    segmented[i][j] = k
                    break

    segmented = segmented.astype(np.uint8)
    original = imread(image_path)
    output5 = mark_boundaries(original, segmented, color=(1, 1, 1))

    return meanImage , output5


image_path = "101085.jpg"
segment = 1000
k=2

labels1 , output1 = feature1ToPlot(k=k)
labels2, output2 = feature2ToPlot(k=k)
labels3, superpixels3 , output3 = feature3ToPlot(segment=segment,k=k)
labels4 , output4 = feature4ToPlot(segment=segment,k=k)
labels5 , output5= feature5ToPlot(segment=segment,k=k)


plt.figure(figsize=(15, 3))

plt.subplot(1, 5, 1)
plt.imshow(labels1)
plt.title('Image Label 1')
plt.axis('off')

plt.subplot(1, 5, 2)
plt.imshow(labels2)
plt.title('Image Label 2')
plt.axis('off')

plt.subplot(1, 5, 3)
plt.imshow(labels3)
plt.title('Image Label 3')
plt.axis('off')

plt.subplot(1, 5, 4)
plt.imshow(labels4)
plt.title('Image Label 4')
plt.axis('off')

plt.subplot(1, 5, 5)
plt.imshow(labels5)
plt.title('Image Label 5')
plt.axis('off')

plt.show()



