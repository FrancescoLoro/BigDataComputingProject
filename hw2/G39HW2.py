from scipy.spatial import distance
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

    :param S: set of points
    :return: max distance between two points
    """

    max_dist = 0
    for i in S:
        for j in S:
            if i == j:
                continue
            # numpy version: numpy.linalg.norm(a-b)
            curr_dist = distance.euclidean(i, j)
            if max_dist < curr_dist:
                max_dist = curr_dist
    return max_dist


def twoApproxMPD(S, k):
    """
        receives in input a set of points S and an interger k < |S|, selects k points at random from S
        (let S' denote the set of these k points) and returns the maximum distance d(x,y), over all x in
         S' and y in S. Define a constant SEED in your main program (e.g., assigning it one of your
         university IDs as a value), and use that value as a seed for the random generator.
         For Python users: you can use the method random.seed(SEED) from the module random.

    :param S: set of points
    :param k: number of points
    :return: maximum distance over between k points and all S points
    """
    assert k < len(S), "k is less than |S|"
    max_dist = 0
    random.seed(1237770)
    # TODO scegli i punti random senza fare shuffle
    random.shuffle(S)
    # print("{} {}".format(str(S[:k]), str(S[k:])))
    for i in S[:k]:
        for j in S:
            # TODO toglia la radice dalla dist euclideiana e calcola solo alla fine
            max_dist = max(max_dist, distance.euclidean(i, j))
    return max_dist


def kCenterMPD(S, k):
    """
        receives in input a set of points S and an integer k < |S|, and returns a set C of k centers
        selected from S using the Farthest-First Traversal algorithm. It is important that kCenterMPD(S,k)
        run in O(|S|*k) time (see exercise on Slide 23 of the set of Slides on Clustering, Part 2.

    :param S: set of points
    :param k: number of points
    :return: Farthest-First Traversal algorithm result
    """
    assert k < len(S), "k is less than |S|"
    C = [S[0]]
    # C += [argmax_distance(S[1:], C) for i in range(1, k)]
    for i in range(1, k):
        # Find the point ci ∈ S − C that maximizes d(ci, C)
        ci = argmax_distance(S[1:], C)
        C.append(ci)
    return C


def argmax_distance(S, C):
    """
    Find the point si ∈ S − C that maximizes d(si, C)
    :param S:
    :param C:
    :return:
    """
    max_dist = 0
    argmax_dist = None

    for si in S:
        if si in C:
            continue

        cur_dist = math.inf
        for cj in C:
            cur_dist = min(cur_dist, distance.euclidean(si, cj))
        # print("  checking d({}, {}) = {} vs {} with max_dist = {}".format(si, C, cur_dist, argmax_dist, max_dist))
        if max_dist < cur_dist:
            # print("  argmax is {}".format(si))
            max_dist = cur_dist
            argmax_dist = si
    # print(" return {}".format(argmax_dist))
    return argmax_dist


if __name__ == "__main__":

    # Check cmd line param, spark setup
    assert len(sys.argv) == 3, "Usage: python G39HW2.py <K> <path-to-file>"
    # Read number of partitions
    K = sys.argv[1]
    assert K.isdigit(), "K must be an integer"
    K = int(K)
    # Read input file and subdivide it into K random partitions
    data_path = sys.argv[2]
    assert os.path.isfile(data_path), "File or folder not found"
    points = readTuplesSeq(data_path)

    # print("\nEXACT ALGORITHM")
    # start = timeit.default_timer()
    # print("Max distance = {}".format(exactMPD(points)))
    # stop = timeit.default_timer()
    # print("Running time = {}".format(stop - start))

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
