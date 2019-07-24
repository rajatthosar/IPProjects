import Task3.EdgeDetector as ed
import numpy as np
import cv2
import time

H_point = [
    [[-1, -1, -1],
     [-1, -1, -1],
     [-1, -1, -1]],

    [[-1, -1, -1],
     [-1, 26, -1],
     [-1, -1, -1]],

    [[-1, -1, -1],
     [-1, -1, -1],
     [-1, -1, -1]]
]


def GetAccumulatorValues(height, width, r_max, binarizedImage):
    """This function generates the voting for accumulator and returns the voted accumulator matrix"""

    ABRAccumulator = np.zeros([width, height, r_max])

    for x in range(len(binarizedImage)):
        for y in range(len(binarizedImage[0])):
            if binarizedImage[x][y] == 0:
                continue
            for a in range(x - 28, x + 28):
                if a < 0 or a >= width:
                    continue
                for b in range(y - 28, y + 28):
                    if b < 0 or b >= height:
                        continue
                    r = int(round(np.sqrt((x - a)** 2 + (y - b)** 2)))
                    if 18 < r < 28:
                        ABRAccumulator[a][b][r - 19] += 1

    return ABRAccumulator


def Convolve3D(ABRAccumulator, H_point):
    offset = int(len(H_point) / 2)
    G = np.zeros(ABRAccumulator.shape)

    for x in range(offset, ABRAccumulator.shape[0] - offset):

        for y in range(offset, ABRAccumulator.shape[1] - offset):
            for z in range(offset, ABRAccumulator.shape[2] - offset):
                if ABRAccumulator[x][y][z] == 1:
                    continue
                G[x, y, z] = np.sum(
                    np.multiply(
                        ABRAccumulator[x - offset:x + offset + 1, y - offset:y + offset + 1, z - offset:z + offset + 1],
                        H_point))

    return G, np.max(G), np.min(G)


def GetHoughSpace(ABRAccumulator, H_point):
    rho_peaks, maximum, minimum = Convolve3D(ABRAccumulator, H_point)
    normalized_peaks = ed.Normalize(rho_peaks, minimum, maximum)
    normalized_peaks[normalized_peaks <= 150] = 0
    normalized_peaks[normalized_peaks > 150] = 255
    a, b, r = np.where(normalized_peaks == 255)

    return a, b, r


def DrawCircles(a, b, radii, ImGreens, binarizedImage):
    for x in range(ImGreens.shape[0]):
        iter_start = time.time()

        for y in range(ImGreens.shape[1]):
            if binarizedImage[x, y] == 0:
                continue
            for aInstance, bInstance, r in zip(a, b, radii):
                cv2.circle(ImGreens, (aInstance, bInstance), r + 19, (0, 255, 0), 1)

        print("Time taken by the iteration: ", time.time() - iter_start)
    cv2.imwrite("OutputData\\coin.jpg", ImGreens)


def Task3BonusHandler():
    binarizedImage = ed.DetectEdges(0.3)
    colored_image = cv2.imread("InputData\\original_imgs\\hough.jpg")
    gray_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2GRAY)
    ImGreens = np.dstack((gray_image, gray_image, gray_image))
    width, height = binarizedImage.shape
    r_max = 9
    ABRAccumulator = GetAccumulatorValues(height, width, r_max, binarizedImage)
    a, b, radius = GetHoughSpace(ABRAccumulator, H_point)
    DrawCircles(a, b, radius, ImGreens, binarizedImage)
    print("Task3bonus executed successfully!")
