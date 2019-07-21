import cv2
import sys
import numpy as np


def GetSIFTKeyPointsAndDescriptors(path):
    """
    This function calculates the SIFT keypoints and
    the descriptors for the image present in the path.
    It returns the image (in grayscale), the descriptors
    and a list of keypoints

    :param path: path of the stored image
    :return: image {Type: numpy.ndarray},
             keypoint descriptors {Type: numpy.ndarray},
             keypoints {Type: List}
    """

    filename, delim, extension = path.rpartition('.')

    # This block is implemented to distinct names
    # to the generated outputs
    if 'left' in filename:
        number = '1'
    elif 'right' in filename:
        number = '2'
    else:
        number = filename[len(filename) - 1]

    img = cv2.imread(path)

    # Instantiate SIFT object
    sift = cv2.xfeatures2d.SIFT_create()
    keyp, desc = sift.detectAndCompute(img, None)
    keyed_img = cv2.drawKeypoints(img, keyp, None)

    # This block is used to determine which function called the
    # current function. This helps in placing the outputs in the
    # correct directories.
    source = sys._getframe().f_back.f_code.co_name
    if "Task1" in source:
        cv2.imwrite('.\\ProjectOutputs\\task1\\' + 'sift' + number + '.jpg', keyed_img)
    elif "Task2" in sys._getframe().f_back.f_code.co_name:
        cv2.imwrite('.\\ProjectOutputs\\task2\\' + 'sift' + number + '.jpg', keyed_img)
    else:
        print("Warning! Method is being accessed by external source: ", source)

    return img, desc, keyp


def matchFeatures(img1, des1, key1, img2, des2, key2):
    """
    This function matches the keypoints from image at img1 to image at img2

    :param img1: First image matrix
    :param des1: Descriptors of Keypoints for img1
    :param key1: Keypoints for img1
    :param img2: Second image matrix
    :param des2: Descriptors of Keypoints for img2
    :param key2: Keypoints for img2
    :return: matched features {Type: List},
             knn matched image {Type: numpy.ndarray}
             points1 {Type: List}
             points1 {Type: List}
             points2 {Type: List}
    """

    bruteForceMatcher = cv2.BFMatcher()
    matches = bruteForceMatcher.knnMatch(des1, des2, 2)
    matchedFeatures = []

    points1 = []
    points2 = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            matchedFeatures.append(m)
            # points1.append(key1[m.trainIdx].pt)
            # points2.append(key2[m.trainIdx].pt)

    img3 = cv2.drawMatches(img1, key1, img2, key2, matchedFeatures, None, flags=2)

    # This block is used to determine which function called the
    # current function. This helps in placing the outputs in the
    # correct directories.
    source = sys._getframe().f_back.f_code.co_name
    if "Task1" in source:
        cv2.imwrite('.\\ProjectOutputs\\task1\\' + 'matches_knn.jpg', img3)
    elif "Task2" in sys._getframe().f_back.f_code.co_name:
        cv2.imwrite('.\\ProjectOutputs\\task2\\' + 'matches_knn.jpg', img3)
    else:
        print("Warning! Method is being accessed by external source: ", source)

    return matchedFeatures, img3, points1, points2


def CalculateEuclideanDistance(point1, point2):
    """
    This function returns the Euclidean distance between point1 and point2

    :param point1: First point
    :param point2: Second point
    :return: Euclidean Distance {Type: Float}
    """

    return np.sqrt(np.sum((point1 - point2) ** 2, axis=1))


def ClassifyDataPoints(data, centroids):
    """
    This function splits the datapoints by the virtue
    of the centroids closest to them

    :param data: Points being classified
    :param centroids: The cluster centers around which classification
            takes place
    :return: Centroid Data {Type: List}
    """

    x = []
    y = []
    CentroidData = [[0] * len(centroids)]
    CentroidData = np.transpose(CentroidData).tolist()

    for instance in data:
        distances = CalculateEuclideanDistance(centroids, instance)
        CentroidData[distances.argmin()].append(instance)

    for i in range(len(CentroidData)):
        CentroidData[i].pop(0)

    return CentroidData


def RecalculateCentroids(CentroidData):
    """This function updates the Cluster Centroids from
    the newly classified data"""

    x = []
    y = []
    z = []
    for i in range(len(CentroidData)):
        points = np.array(CentroidData[i])
        # print(points)
        try:
            x.append(np.mean(points[:, 0]))
            y.append(np.mean(points[:, 1]))
            z.append(np.mean(points[:, 2]))
        except IndexError:
            x.append(0)
            y.append(0)
            z.append(0)

    data = np.array([np.asarray(x), np.asarray(y), np.asarray(z)])
    updatedCentroids = np.transpose(data)
    x = []
    y = []
    z = []
    return updatedCentroids
