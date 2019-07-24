import numpy as np
import cv2

# Sobel y-direction kernel
Hy = [
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
]

# Sobel x-direction kernel
Hx = [
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
]


def GetImageData(path, isPadding):
    """Preprocesses the data from the image file"""

    # Read the image in grayscale
    file = cv2.imread(path, 0)
    imgdata = []

    for row in file:
        if not isPadding:
            imgdata.append(row)
        else:
            zeroPaddedRow = []
            zeroPaddedRow.insert(0, 0)  # zero padding to the left of the leftmost column

            # Additional loop: to break the list "row" into its elements
            # Failing to add this loop would result in a list of lists-
            # -in zeroPaddedRow, which is undesirable.
            for item in row:
                zeroPaddedRow.append(item)
            zeroPaddedRow.append(0)  # zero padding to the right of the rightmost column
            imgdata.append(zeroPaddedRow)
    if isPadding:
        zeroRow = []

        # Created a row of zeros to be padded at the top and bottom of the imgdata list
        for zeros in range(len(imgdata[0])):
            zeroRow.append(0)
        imgdata.insert(0, zeroRow)
        imgdata.append(zeroRow)
    return imgdata  # returns zero padded imgdata list of lists


def InitializeG(rows, cols):
    """Initializes a container to store convolution results"""
    G = []
    tempRow = [0] * (cols - 2)
    for row in range(rows - 2):
        G.append(tempRow)
    return np.asarray(G)


def Convolve(imageData, H):
    """Performs Convolution operation of Filter H with image matrix imageData"""
    u = len(H)
    v = len(H[0])
    G = InitializeG(len(imageData), len(imageData[0]))

    min_value = 0
    max_value = 0

    # max_offset is used to compensate for the negative values in the matrices
    max_offset = u // 2

    for index_i in range(1, len(imageData) - max_offset):
        for index_j in range(1, len(imageData[0]) - max_offset):
            innerSum = 0
            for index_u in range(u):
                for index_v in range(v):
                    innerSum += H[index_u][index_v] * imageData[index_i + index_u - max_offset][
                        index_j + index_v - max_offset]
                    G[index_i - 1][index_j - 1] = innerSum
            if (innerSum < min_value):
                min_value = innerSum
            elif (innerSum > max_value):
                max_value = innerSum
    Gnorm = np.abs(G) / np.max(np.abs(G))
    return G, Gnorm, max_value, min_value


def NormalizeImage(G, minimum, maximum):
    """Displays the image for the matrix input at first parameter.
    Requires the minimum and maximum of the matrix"""

    # Removing the negative values
    G = (G - minimum) / (maximum - minimum)
    return G


def Normalize(G, minimum, maximum):
    """Displays the image for the matrix input at first parameter.
    Requires the minimum and maximum of the matrix"""

    # Removing the negative values
    G = (G - minimum) / (maximum - minimum)
    return G * 255


def BinarizeImage(image, binThresh=0.15):
    """Binarize the given image"""
    for row in range(len(image)):
        for pixel in range(len(image[0])):
            if image[row][pixel] > binThresh:
                image[row][pixel] = 255
            else:
                image[row][pixel] = 0
    return image


def DetectEdges(binThresh=0.15):
    filepath = "InputData\\original_imgs\\hough.jpg"
    img = GetImageData(filepath, True)
    Gx, _, Xmaximum, Xminimum = Convolve(img, Hx)
    Gy, _, Ymaximum, Yminimum = Convolve(img, Hy)
    G = np.sqrt(np.multiply(Gx, Gx) + np.multiply(Gy, Gy))
    normalizedEdges = NormalizeImage(G, np.min(G), np.max(G))
    binarizedImage = BinarizeImage(normalizedEdges, binThresh)

    return binarizedImage
