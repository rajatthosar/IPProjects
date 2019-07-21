import numpy as np
import matplotlib.pyplot as plt
import Functions as myFunc

# Created a separate python file for frequently
# accessed functions
UBITName = 'rthosar'
np.random.seed(sum([ord(c) for c in UBITName]))


def PlotScatters(X, DataPoints0, DataPoints1, DataPoints2, Centroids, label):
    """
    Create Scatter Plots of the given DataPoints with Centroids

    :param X: Original Data points
    :param DataPoints0: Individual datapoints for subtask1
    :param DataPoints1: Individual datapoints for subtask2
    :param DataPoints2: Individual datapoints for subtask3
    :param Centroids: Cluster centers
    :param label: A name to distinguish the plots
    :return: nothing
    """
    plt.scatter(DataPoints0[:, 0], DataPoints0[:, 1], marker='^', c=(1, 0, 0))
    for Xindex in X:
        plt.text(Xindex[0], Xindex[1], "(" + str(Xindex[0]) + ", " + str(Xindex[1]) + ")")
    for Xindex in Centroids:
        plt.text(Xindex[0], Xindex[1], "(" + str(np.around(Xindex[0], 1)) + ", " + str(np.around(Xindex[1], 1)) + ")")
    plt.scatter(DataPoints1[:, 0], DataPoints1[:, 1], marker='^', c=(0, 1, 0))
    plt.scatter(DataPoints2[:, 0], DataPoints2[:, 1], marker='^', c=(0, 0, 1))
    plt.scatter(Centroids[:, 0], Centroids[:, 1], marker='o', c=((1, 0, 0), (0, 1, 0), (0, 0, 1)))

    plt.savefig('.\\ProjectOutputs\\task3\\iter' + label + '.jpg')
    plt.clf()


def Task33Handler():
    """
    This function acts as the entry point to the code in this .py file

    :return: nothing
    """
    X = np.array([[5.9, 3.2], [4.6, 2.9], [6.2, 2.8], [4.7, 3.2], [5.5, 4.2],
                  [5.0, 3.0], [4.9, 3.1], [6.7, 3.1], [5.1, 3.8], [6.0, 3.0]])
    Centroids = np.array([[6.2, 3.2], [6.6, 3.7], [6.5, 3.0]])

    CentData = myFunc.ClassifyDataPoints(X, Centroids)
    newCentroids = myFunc.RecalculateCentroids(CentData)
    DataPoints0 = np.array(CentData[0])
    DataPoints1 = np.array(CentData[1])
    DataPoints2 = np.array(CentData[2])

    # ------------------------Task 3.1 Plot ----------------------------------------
    PlotScatters(X, DataPoints0, DataPoints1, DataPoints2, Centroids, '1_a')
    # ------------------------Task 3.1 Plot ----------------------------------------

    # ------------------------Task 3.2 Plot ----------------------------------------
    PlotScatters(X, DataPoints0, DataPoints1, DataPoints2, newCentroids, '1_b')
    # ------------------------Task 3.2 Plot ----------------------------------------

    CentData = myFunc.ClassifyDataPoints(X, newCentroids)
    DataPoints0 = np.array(CentData[0])
    DataPoints1 = np.array(CentData[1])
    DataPoints2 = np.array(CentData[2])

    # ------------------------Task 3.3 Plot A ----------------------------------------
    PlotScatters(X, DataPoints0, DataPoints1, DataPoints2, newCentroids, '2_a')
    # ------------------------Task 3.3 Plot A ----------------------------------------

    newCentroids = myFunc.RecalculateCentroids(CentData)

    # ------------------------Task 3.3 Plot B ----------------------------------------
    PlotScatters(X, DataPoints0, DataPoints1, DataPoints2, newCentroids, '2_b')
    # ------------------------Task 3.3 Plot B ----------------------------------------

    print("Task 3 finished successfully!\n")
