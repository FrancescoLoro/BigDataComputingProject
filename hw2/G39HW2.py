from scipy.spatial import distance
import math
import random


def exactMPD(S):
    """
        receives in input a set of points S and returns the max distance between two points in S.

    :param S:
    :return:
    """
    max_dist = 0
    for i in S:
        for j in S:
            if i == j: continue
            curr_dist = distance.euclidean(i, j)   # numpy version: numpy.linalg.norm(a-b)
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

    :param S:
    :param k:
    :return:
    """
    max_dist = 0
    random.seed(666)
    random.shuffle(S)
    # print("{} {}".format(str(S[:k]), str(S[k:])))
    for i in S[:k]:
        for j in S:
            max_dist = max(max_dist, distance.euclidean(i, j))
    return max_dist


def kCenterMPD(S, k):
    """
        receives in input a set of points S and an integer k < |S|, and returns a set C of k centers
        selected from S using the Farthest-First Traversal algorithm. It is important that kCenterMPD(S,k)
        run in O(|S|*k) time (see exercise on Slide 23 of the set of Slides on Clustering, Part 2.

    :param S:
    :param k:
    :return:
    """
    assert k < len(S)
    C = [S[0]]
    # C += [argmax_distance(S[1:], C) for i in range(1, k)]
    for i in range(1, k):
        ci = argmax_distance(S[1:], C)  # Find the point ci ∈ S − C that maximizes d(ci, C)
        C.append(ci)
    return C



def argmax_distance(S, C):
    """
    Find the point ci ∈ S − C that maximizes d(ci, C)
    :param S:
    :param C:
    :return:
    """
    max_dist = 0
    argmax_dist = None
    # print("{} {}".format(str(S[:k]), str(S[k:])))
    for ci in S:
        if ci in C:
            continue

        cur_dist = math.inf
        for j in C:
            cur_dist = min(cur_dist, distance.euclidean(ci, j))
        print("  checking d({}, {}) = {} vs {} with max_dist = {}".format(ci, C, cur_dist, argmax_dist, max_dist))
        if max_dist < cur_dist:
            print("  argmax is {}".format(ci))
            max_dist = cur_dist
            argmax_dist = ci
    print(" return {}".format(argmax_dist))
    return argmax_dist


if __name__ == "__main__":
    S1 = [(0, 0), (10, 10), (1, 1), (5, 5), (10, 0)]
    S2 = [(0, 0)]
    S3 = [0, 10, 1, 5, 10]

    assert exactMPD(S1) == math.sqrt(200)
    assert exactMPD(S2) == 0
    assert exactMPD(S3) == 10

    random.seed(5)
    print("Random: {}".format(random.uniform(1, 20)))
    random.seed(5)
    print("Random: {}".format(random.uniform(1, 20)))
    random.seed(5)
    print("Random: {}".format(random.uniform(1, 20)))

    print("max = {} ".format(twoApproxMPD(S1, 3)))
    print("max = {} ".format(twoApproxMPD(S1, 3)))
    print("max = {} ".format(twoApproxMPD(S1, 3)))
    print("max = {} ".format(twoApproxMPD(S1, 3)))

    print("C: {} in {}".format(kCenterMPD(S1, 2), S1))
    print("C: {} in {}".format(kCenterMPD(S1, 3), S1))
    print("C: {} in {}".format(kCenterMPD(S1, 4), S1))
    print("C: {} in {}".format(kCenterMPD(S3, 2), S3))
