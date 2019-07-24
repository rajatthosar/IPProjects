import numpy as np
import cv2


def Dilate(img, mask):
    """This function dilates the image passed in the parameter with the mask."""

    dilatedImg = np.copy(img)
    for x in range(len(img)):
        for y in range(len(img[0])):
            if img[x][y] == mask[1][1]:
                dilatedImg[x][y] = mask[1][1]
                if x > 0:
                    dilatedImg[x - 1][y] = mask[0][1]
                if x < len(img) - 1:
                    dilatedImg[x + 1][y] = mask[2][1]
                if y < len(img[0]) - 1:
                    dilatedImg[x][y + 1] = mask[1][2]
                if y > 0:
                    dilatedImg[x][y - 1] = mask[1][0]
    return dilatedImg


def Erode(image, mask):
    """This function erodes the image passed in the parameters with the mask"""

    erodedImage = np.copy(image)
    for row in range(len(image)):
        for col in range(len(image[0])):
            if image[row - 1, col] == mask[1][0] and image[row, col - 1] == mask[0][1] and image[
                row, col] == \
                    mask[1][1] and image[row, col + 1] == mask[1][2] and image[row + 1, col] == mask[2][1]:
                # erodedImage[row - 1: row + 2, col] = 0
                # erodedImage[row, col - 1: col + 2] = 0
                erodedImage[row, col] = 255
            else:
                erodedImage[row - 1: row + 2, col] = 0
                erodedImage[row, col - 1: col + 2] = 0
                erodedImage[row, col] = 0
    return erodedImage


def PerformOpening(image, mask):
    """This function performs morphological opening over the given image."""

    erodedImage = Erode(image, mask)
    dilatedImage = Dilate(erodedImage, mask)

    return dilatedImage


def PerformClosing(image, mask):
    """This function performs morphological closing over the given image."""

    dilatedImage = Dilate(image, mask)
    erodedImage = Erode(dilatedImage, mask)

    return erodedImage


def GetBoundaries(image, mask):
    """This function gets the boundaries of the image by eroding it and subtracting from itself"""

    return image - Erode(image, mask)


def Task1Handler():
    """This function works as an entry point to this program.
    It calls all the relevant functions internally to execute
    the code written in this file."""

    imgPath = "InputData\\original_imgs\\noise.jpg"
    img = cv2.imread(imgPath, 0)
    #
    mask = [[0, 255, 0], [255, 255, 255], [0, 255, 0]]
    mask1 = [[255, 255, 255], [255, 255, 255], [255, 255, 255]]

    dilated = Dilate(img, mask1)
    eroded = Erode(dilated, mask1)
    cv2.imwrite("OutputData\\res_noise0.jpg", eroded)

    openedImage = PerformOpening(img, mask1)
    closedImage = PerformClosing(openedImage, mask1)
    cv2.imwrite("OutputData\\res_noise1.jpg", closedImage)
    boundedImage1 = GetBoundaries(closedImage, mask1)
    cv2.imwrite("OutputData\\res_bound1.jpg", boundedImage1)

    closedImage = PerformClosing(img, mask1)
    openedImage = PerformOpening(closedImage, mask1)
    cv2.imwrite("OutputData\\res_noise2.jpg", openedImage)
    boundedImage2 = GetBoundaries(closedImage, mask1)
    cv2.imwrite("OutputData\\res_bound2.jpg", boundedImage2)
    print("Task1 executed successfully!")

