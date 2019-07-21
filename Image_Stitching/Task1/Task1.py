# References:
# https://docs.opencv.org/3.0-beta/modules/cudawarping/doc/warping.html
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
# https://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/

import cv2
import numpy as np

# Created a separate python file for frequently
# accessed functions
import Functions as myFunc


# Created a random seed to reproduce the same output
UBITName = 'rthosar'
np.random.seed(sum([ord(c) for c in UBITName]))


def GetHomographyAndMatchImg(img1, keypt1, img2, keypt2, matchedFeatures, numOfPoints=0):
    """
    This function finds the homography matrix

    :param img1: First image
    :param keypt1: Keypoints for img1
    :param img2: Second image
    :param keypt2: Keypoints for img2
    :param matchedFeatures: The matched features for img1 and img2
           received from matchFeatures(img1, des1, key1, img2, des2, key2)
           function.
    :param numOfPoints: Number of points to match
    :return: Homography matrix {Type: numpy.ndarray}
    """

    selectKP = np.random.choice(matchedFeatures, 10)
    source = np.float32([keypt2[m.queryIdx].pt for m in selectKP]).reshape(-1, 1, 2)
    destination = np.float32([keypt1[m.trainIdx].pt for m in selectKP]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(source, destination, cv2.RANSAC, 5.0)

    if numOfPoints == 0:
        print("Homography matrix:")
        print(H)
    else:
        if len(matchedFeatures) > numOfPoints:
            height, width, ch = img1.shape
            points = np.float32([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]]).reshape(-1, 1, 2)

        else:
            print("Not enough matches are found - %d/%d" % (len(matchedFeatures), numOfPoints))

        draw_params = dict(matchColor=(0, 0, 255),  # draw matches in red color
                           singlePointColor=None,
                           matchesMask=None,  # draw only inliers
                           flags=2)

        img3 = cv2.drawMatches(img1, keypt1, img2, keypt2, selectKP, None, **draw_params)

        cv2.imwrite('.\\ProjectOutputs\\task1\\' + 'matches.jpg', img3)
    return H


def WarpImages(img1, img2, H):
    """
    This function warps the input image with
    the Homography matrix H provided in the arguments

    :param img1: First Image
    :param img2: Second Image
    :param H: Homography matrix to warp the images
    :return: nothing
    """

    warped_img = cv2.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))
    stitcher = cv2.createStitcher(try_use_gpu=True)
    stitched_image = stitcher.stitch((img1, warped_img, img2))
    cv2.imwrite('.\\ProjectOutputs\\task1\\' + 'pano.jpg', stitched_image[1])


def Task1Handler():
    """
    This function acts as an entry point to the code in this .py file

    :return: nothing
    """
    image1, desc1, keypt1 = myFunc.GetSIFTKeyPointsAndDescriptors('.\\proj2_data\\data\\mountain1.jpg')
    # cv2.imshow('asdf',image1)
    image2, desc2, keypt2 = myFunc.GetSIFTKeyPointsAndDescriptors('.\\proj2_data\\data\\mountain2.jpg')
    # cv2.imshow('asdfasdf',image2)

    matchedFeat, joint_img, _, _ = myFunc.matchFeatures(image1, desc1, keypt1, image2, desc2, keypt2)
    # cv2.imshow('asdfasdf',joint_img)

    H = GetHomographyAndMatchImg(image1, keypt1, image2, keypt2, matchedFeat)
    H1 = GetHomographyAndMatchImg(image1, keypt1, image2, keypt2, matchedFeat, numOfPoints=10)
    WarpImages(image1, image2, H)

    print("Task 1 finished successfully!\n")
