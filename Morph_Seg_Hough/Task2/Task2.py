import numpy as np
import cv2
from matplotlib import pyplot as plt
import Task3.EdgeDetector as ed

H_point = [
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
]


def DetectPoint(img, H_point):
    G, _, maximum, minimum = ed.Convolve(img, H_point)
    G = np.abs(G)
    Normed_G = ed.Normalize(G, minimum, maximum)
    NormThresh = 254

    Normed_G[Normed_G <= NormThresh] = 0
    Normed_G[Normed_G > NormThresh] = 255

    x, y = np.where(Normed_G == 255)
    cv2.putText(Normed_G, str([x[1], y[1]]), (y[1] + 5, x[1] + 5), fontFace=cv2.FONT_ITALIC, fontScale=0.4, color=255,
                thickness=1)
    cv2.putText(Normed_G, str([x[0], y[0]]), (y[0] - 65, x[0] + 5), fontFace=cv2.FONT_ITALIC, fontScale=0.4, color=255,
                thickness=1)
    cv2.imwrite("OutputData\\res_point.jpg", Normed_G)


def SegmentImage(image):

    NormThresh = 204

    image[image <= NormThresh] = 0
    image[image > NormThresh] = 255

    colored_image = np.dstack((image, image, image))

    x, y = np.where(image == 255)

    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)

    bounded_image = cv2.rectangle(colored_image, (ymin, xmin), (ymax, xmax), (0, 255, 0))

    cv2.imwrite("OutputData\\res_segment.jpg", bounded_image)


def GetHistogram(image):
    hist = np.zeros(256)

    for row in image:
        for pixel in row:
            hist[pixel] += 1
    hist[0] = 0

    plt.bar(range(len(hist)), hist)
    plt.show()


def Task2Handler():
    img = cv2.imread("InputData\\original_imgs\\point.jpg", 0)
    DetectPoint(img, H_point)

    segimg = cv2.imread("InputData\\original_imgs\\segment.jpg", 0)
    GetHistogram(segimg)
    SegmentImage(segimg)
    print("Task2 executed successfully!")
