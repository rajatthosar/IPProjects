# References:
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_depthmap/py_depthmap.html
# https://stackoverflow.com/questions/1735025/how-to-normalize-a-numpy-array-to-within-a-certain-range


import cv2
import numpy as np

# Created a separate python file for frequently
# accessed functions
import Functions as myFunc


# Created a random seed to reproduce the same output

UBITName = 'rthosar'
np.random.seed(sum([ord(c) for c in UBITName]))


def GetKeyPointsInArray(keypt1, keypt2, matchedFeat):
    """
    This function converts the keypoints to numpy integers

    :param keypt1: Keypoints for first image
    :param keypt2: Keypoints for second image
    :param matchedFeat: matched features for first and second images
    :return: point1 {Type: numpy.ndarray},
             point2 {Type: numpy.ndarray}
    """

    point1 = np.float32([keypt1[m.queryIdx].pt for m in matchedFeat])
    point2 = np.float32([keypt2[m.trainIdx].pt for m in matchedFeat])

    return np.int32(point1), np.int32(point2)


def GetFundamentalMatrix(points1, keypt1, points2, keypt2, matchedFeatures):
    """
    This function returns the Fundamental matrix from the input co-ordinates

    :param points1: Points 1 array obtained from GetKeyPointsInArray(keypt1, keypt2, matchedFeat)
    :param keypt1: Keypoints for image 1
    :param points2: Points 2 array obtained from GetKeyPointsInArray(keypt1, keypt2, matchedFeat)
    :param keypt2: Keypoints for image 2
    :param matchedFeatures: matched features for first and second images
    :return: Fundamental Matrix {Type: numpy.ndarray},
             Selected Points1 {Type: numpy.ndarray},
             Selected Points2 {Type: numpy.ndarray}
    """

    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)
    print("Fundamental Matrix:")
    print(F)
    np.random.shuffle(matchedFeatures)
    points1 = np.int32([keypt1[m.queryIdx].pt for m in matchedFeatures])
    points2 = np.int32([keypt2[m.trainIdx].pt for m in matchedFeatures])

    # inliers = matchedFeatures[mask.ravel()==1]
    selectedPoints1 = points1[mask.ravel() == 1][:10]
    selectedPoints2 = points2[mask.ravel() == 1][:10]

    return F, selectedPoints1, selectedPoints2


def DrawEpiLines(img1, img2, lines_left, lines_right, points1, points2):
    """
    This function draws the EpiLines calculated from the GetEpiLines function

    :param img1: First Image
    :param img2: Second Image
    :param lines_left: Epipolar lines from points2
    :param lines_right: Epipolar lines from points1
    :param points1: Points 1 array obtained from GetKeyPointsInArray(keypt1, keypt2, matchedFeat)
    :param points2: Points 2 array obtained from GetKeyPointsInArray(keypt1, keypt2, matchedFeat)
    :return: Epipolar line drawn image1 {Type:numpy.ndarray},
             Epipolar line drawn image2 {Type:numpy.ndarray}
    """

    row, cols, ch = img1.shape
    for left, right, point_1, point_2 in zip(lines_left, lines_right, points1, points2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0_left, y0_left = map(int, [0, -left[2] / left[1]])
        x1_left, y1_left = map(int, [cols, -(left[2] + left[0] * cols) / left[1]])
        x0_right, y0_right = map(int, [0, -right[2] / right[1]])
        x1_right, y1_right = map(int, [cols, -(right[2] + right[0] * cols) / right[1]])
        img1 = cv2.line(img1, (x0_left, y0_left), (x1_left, y1_left), color, 1)
        img2 = cv2.line(img2, (x0_right, y0_right), (x1_right, y1_right), color, 1)
        img1 = cv2.circle(img1, tuple(point_1), 7, color, -1)
        img2 = cv2.circle(img2, tuple(point_2), 7, color, -1)
    return img1, img2


def GetEpiLines(img1, img2, points1, points2, F):
    """
    This function generates Epipolar lines

    :param img1: First Image
    :param img2: Second Image
    :param points1: Points 1 array obtained from GetKeyPointsInArray(keypt1, keypt2, matchedFeat)
    :param points2: Points 2 array obtained from GetKeyPointsInArray(keypt1, keypt2, matchedFeat)
    :param F: Fundamental Matrix
    :return: nothing
    """
    lines1 = cv2.computeCorrespondEpilines(points2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    # lined_img1, lined_img2 = DrawEpiLines(img1, img2, lines1, points1, points2)

    lines2 = cv2.computeCorrespondEpilines(points1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    epilined_img1, epilined_img2 = DrawEpiLines(img1, img2, lines1, lines2, points1, points2)

    cv2.imwrite('.\\ProjectOutputs\\task2\\epi_left.jpg', epilined_img1)
    cv2.imwrite('.\\ProjectOutputs\\task2\\epi_right.jpg', epilined_img2)


def ComputeDisparityMap(img1, img2):
    """
    This function calculates the Disparity Map
    between the two images passed in the arguments"
    
    :param img1: First Image
    :param img2: Second Image
    :return: nothing
    """""

    stereo = cv2.StereoBM_create(64, 21)
    disparity_map = stereo.compute(img1, img2).astype(np.float32)
    maximum = np.max(disparity_map)
    minimum = np.min(disparity_map)
    disparity_map -= minimum
    disparity_map *= (255/maximum)
    cv2.imwrite('.\\ProjectOutputs\\task2\\disparity.jpg', disparity_map)



def Task2Handler():
    """
    This function acts as the entry point to this .py file

    :return: nothing
    """
    image1, desc1, keypt1 = myFunc.GetSIFTKeyPointsAndDescriptors('.\\proj2_data\\data\\tsucuba_left.png')
    image2, desc2, keypt2 = myFunc.GetSIFTKeyPointsAndDescriptors('.\\proj2_data\\data\\tsucuba_right.png')
    matchedFeat, joint_img, _, _ = myFunc.matchFeatures(image1, desc1, keypt1, image2, desc2, keypt2)
    points1, points2 = GetKeyPointsInArray(keypt1, keypt2, matchedFeat)
    F, inliers1, inliers2 = GetFundamentalMatrix(points1, keypt1, points2, keypt2, matchedFeat)
    GetEpiLines(image1, image2, inliers1, inliers2, F)
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    ComputeDisparityMap(gray_image1, gray_image2)

    print("Task 2 finished successfully!\n")