import Task3.EdgeDetector as ed
import numpy as np
import cv2

angles = [a for a in range(-89, 91)]
anglesinRad = np.deg2rad(angles)
sin = np.sin(anglesinRad).tolist()
cos = np.cos(anglesinRad).tolist()

H_point = [
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1],
    [-1, -1, 24, -1, -1],
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1]
]


def GetAccumulatorValues(rho_max, angles, binarizedImage):
    """This function generates the voting for accumulator and returns the voted accumulator matrix"""

    RhoThetaAccumulator = np.zeros([2 * rho_max + 1, len(angles)])

    for x in range(len(binarizedImage)):
        for y in range(len(binarizedImage[0])):
            for angle in angles:
                if binarizedImage[x][y] == 0:
                    continue
                theta = angle + 89
                rho = int(y * cos[theta] + x * sin[theta])

                # Compensation for -ve values of rhos
                rho += rho_max
                RhoThetaAccumulator[rho, theta] += 1
    return RhoThetaAccumulator


def GetHoughSpace(RhoThetaAccumulator, H_point):
    """This function runs the convolution of accumulator with point filter and generates the parameter space variables"""

    rho_peaks, _, maximum, minimum = ed.Convolve(RhoThetaAccumulator, H_point)
    normalized_peaks = ed.Normalize(rho_peaks, minimum, maximum)
    cv2.imwrite("OutputData\\hough_sin_pre.jpg", normalized_peaks)
    # rho_peaks, _, maximum, minimum = ed.Convolve(normalized_peaks, H_point)
    # normalized2_peaks = ed.Normalize(rho_peaks, minimum, maximum)
    # cv2.imwrite("hough_sinusoids.jpg", normalized2_peaks)

    normalized_peaks[normalized_peaks <= 70] = 0
    normalized_peaks[normalized_peaks > 70] = 255
    cv2.imwrite("OutputData\\norm_thresh.jpg", normalized_peaks)
    rhos, thetas = np.where(normalized_peaks == 255)

    return rhos, thetas


def DrawLines(rhos, thetas, rho_max, ImReds, ImBlues, binarizedImage):
    """This function is used to draw the detected lines."""

    count_reds = 0
    count_blues = 0

    for x_line in range(ImReds.shape[0]):
        for y_line in range(ImReds.shape[1]):
            if binarizedImage[x_line, y_line] == 0:
                continue
            for rho, theta in zip(rhos, thetas):
                rho_calc = round((y_line * cos[theta] + x_line * sin[theta]), 0)
                rm = rho - rho_max

                if round(rho_calc) == rm:
                    if 84 < theta < 89:
                        ImReds[x_line, y_line] = np.array([0, 0, 255])
                        count_reds += 1
                    elif 50 < theta < 55:
                        ImBlues[x_line, y_line] = np.array([255, 0, 0])
                        count_blues += 1
                    else:
                        pass

    cv2.imwrite("OutputData\\red_lines.jpg", ImReds)
    cv2.imwrite("OutputData\\blue_lines.jpg", ImBlues)
    print("Number of Red lines drawn: ", count_reds)
    print("Number of Blue lines drawn: ", count_blues)


def Task3Handler():
    """This function serves as the entry point to this program and calls all of the functions internally"""

    binarizedImage = ed.DetectEdges(0.12)
    colored_image = cv2.imread("InputData\\original_imgs\\hough.jpg")
    gray_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2GRAY)
    ImReds = np.dstack((gray_image, gray_image, gray_image))
    ImBlues = ImReds.copy()
    width, height = binarizedImage.shape
    rho_max = int(np.ceil(np.sqrt(width ** 2 + height ** 2)))

    RhoThetaAccumulator = GetAccumulatorValues(rho_max, angles, binarizedImage)
    rhos, thetas = GetHoughSpace(RhoThetaAccumulator, H_point)
    DrawLines(rhos, thetas, rho_max, ImReds, ImBlues, binarizedImage)
    print("Task3 executed successfully!")
