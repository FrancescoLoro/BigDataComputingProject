# from scipy.spatial import distance # distance.euclidean
from tqdm import tqdm
import math
import random
import sys
import os
import timeit


def readTuplesSeq(inputfile):
    points = list()
    f = open(inputfile, "r")
    if f.mode == "r":
        for i in f:
            line = i.split(",")
            t = tuple(float(dim) for dim in line)
            points.append(t)
    f.close()
    return points


def exactMPD(S):
    """
        receives in input a set of points S and returns the max distance between two points in S.

    :param S: a set of points S, each point is a tuple.
    :return: the max distance between two points in S.
    """
    max_dist = 0
    for i in range(len(S)-1):
        for j in range(i+1, len(S)):  # d(S[i],S[j]) = d(S[j],S[i]) therefore skip it.
            curr_dist = quad_distance(S[i], S[j])  # does not execute the sqrt
            if max_dist < curr_dist:
                max_dist = curr_dist
    return math.sqrt(max_dist)


def twoApproxMPD(S, k):
    """
        receives in input a set of points S and an interger k < |S|, selects k points at random from S
        (let S' denote the set of these k points) and returns the maximum distance d(x,y), over all x in
         S' and y in S. Define a constant SEED in your main program (e.g., assigning it one of your
         university IDs as a value), and use that value as a seed for the random generator.
         For Python users: you can use the method random.seed(SEED) from the module random.

    :param S: set of points
    :param k: number of points
    :return: maximum distance over between k points and all S points,
                zero if k is zero, raise ValueError if S has no element
    """
    assert k < len(S), "k >= |S|"
    max_dist = 0
    random.seed(1206597)
    centroids = random.sample(S, k)  # select k random centroids without repetition from S
    for ci in centroids:
        for sj in S:
            cur_dist = quad_distance(ci, sj)  # compare squared distances
            if max_dist < cur_dist:
                max_dist = cur_dist
    return math.sqrt(max_dist)  # root of squared distance == euclidean distance


def kCenterMPD(S, k):
    """
        receives in input a set of points S and an integer k < |S|, and returns a set C of k centers
        selected from S using the Farthest-First Traversal algorithm. It is important that kCenterMPD(S,k)
        run in O(|S|*k) time (see exercise on Slide 23 of the set of Slides on Clustering, Part 2.

    :param S: set of points
    :param k: number of points
    :return: Farthest-First Traversal algorithm result
    """
    assert k < len(S), "k < |S| is needed, but k >= |S| is found"
    random.seed(1206597)
    c0 = random.choice(S)
    C = [c0]                            # first centroid randomly selected
    S_dist = [math.inf for si in S]     # init distances of si from C
    sj_max = c0                         # first element with max distance
    S_dist[0] = 0                       # d(S[0], C) is 0

    for i in range(1, k):  # select other k-1 centroids with FFT algorithm
        max_distance = 0
        j_max = -1

        # Find the point sj ∈ S − C that maximizes d(sj, C)
        for j, sj in enumerate(S):
            if S_dist[j] == 0:              # sj is already in C, skip
                continue
            cur_dist = min(quad_distance(sj, C[-1]), S_dist[j])  # d(sj, C), C[-1] is the last added centroid
            S_dist[j] = cur_dist            # update sj distance from C
            if cur_dist > max_distance:     # check if sj is the farthest element from C
                sj_max = sj                 # save the farthest element
                max_distance = cur_dist     # update the max distance of the element
                j_max = j                   # update the index of the farthest element

        # add the farthest element to centroid and update it's distance from C to 0
        C.append(sj_max)        # add the farthest element to C
        S_dist[j_max] = 0       # set distance of the new centroid from C to 0, cause it has been added to C
    return C


def quad_distance(p1, p2):
    """
    :param p1: a tuple of numbers
    :param p2: a tuple of numbers, with the same length of p1
    :return: squared distance between the points, zero if the tuple have no elements
    """
    assert len(p1) == len(p2), "input points must have the same num of components"
    dist = 0
    for i in range(len(p1)):
        dist += (p1[i] - p2[i])*(p1[i] - p2[i])
    return dist


def euclidean(p1, p2):
    """
    Calculate the euclidean distance between two points
    :param p1: a tuple of numbers
    :param p2: a tuple of numbers, with the same length of p1
    :return: euclidean distance between p1 and p2, zero if p1 and p2 has no elements
    """
    return math.sqrt(quad_distance(p1, p2))


if __name__ == "__main__":

    # Check cmd line param, spark setup
    assert len(sys.argv) == 3, "Usage: python G39HW2.py <K> <path-to-file>"

    K = sys.argv[1]  # Read number of partitions
    assert K.isdigit(), "K must be an integer"
    K = int(K)

    data_path = sys.argv[2]
    assert os.path.isfile(data_path), "File or folder not found"
    points = readTuplesSeq(data_path)  # Read input tuples

    print("\nEXACT ALGORITHM")
    start = timeit.default_timer()
    print("Max distance = {}".format(exactMPD(points)))
    stop = timeit.default_timer()
    print("Running time = {}".format(stop - start))

    print("\n2-APPROXIMATION ALGORITHM")
    print("k = {}".format(K))
    start = timeit.default_timer()
    print("Max distance = {}".format(twoApproxMPD(points, K)))
    stop = timeit.default_timer()
    print("Running time = {}".format(stop - start))

    print("\nk-CENTER-BASED ALGORITHM")
    print("k = {}".format(K))
    start = timeit.default_timer()
    print("Max distance = {}".format(exactMPD(kCenterMPD(points, K))))
    stop = timeit.default_timer()
    print("Running time = {}".format(stop - start))
