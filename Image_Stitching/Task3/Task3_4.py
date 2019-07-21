import numpy as np
import cv2
import Functions as myFunc

# Created a separate python file for frequently
# accessed functions
UBITName = 'rthosar'
np.random.seed(sum([ord(c) for c in UBITName]))
MAX_ITER = 50


def FlattenData(imgData):
    """
    This function converts the multi dimensional array to a
    flattened 1D array

    :param imgData: ndarray of image
    :return: flattened array {Type: List}
    """
    flatImg = []
    for row in imgData:
        for pixel in row:
            flatImg.append(pixel)
    return flatImg


def Task34Handler():
    """
    This function acts as the entry point to the code in this .py file

    :return: nothing
    """
    kList = [3, 5, 10, 20]

    for k in kList:
        image = cv2.imread('.\\proj2_data\\data\\baboon.jpg')
        flatImg = FlattenData(image)

        Centroids = []
        for index in range(k):
            Centroids.append(np.random.randint(0, 255, 3).tolist())

        data = myFunc.ClassifyDataPoints(flatImg, Centroids)
        newCentroids = myFunc.RecalculateCentroids(data)

        for i in range(MAX_ITER):
            print(str(i), "th iteration")
            data = myFunc.ClassifyDataPoints(flatImg, newCentroids)
            oldCentroids = newCentroids
            newCentroids = myFunc.RecalculateCentroids(data)
            if (np.mean(newCentroids - oldCentroids) ** 2 < 0.001):
                break

        out = image.copy()

        for row_index in range(len(out)):
            for col_index in range(len(out[0])):
                distance = myFunc.CalculateEuclideanDistance(out[row_index, col_index], newCentroids)
                out[row_index][col_index] = newCentroids[distance.argmin()]

        cv2.imwrite('..\\ProjectOutputs\\task3\\baboon_' + str(k) + '.jpg', out)
        print("Baboon_"+str(k)+" complete!")
    print("Task 3.4 successfully completed!")
