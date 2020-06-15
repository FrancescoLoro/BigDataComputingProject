# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# METHOD runSequential
# Sequential 2-approximation for diversity maximization based on matching
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# Import Packages
import os
import sys
import random
import math
import timeit

from pyspark import SparkConf, SparkContext, TaskContext

TESTING = True


def squared_euclidean_dist(p, q):
    """
    Compute the squared euclidean distance (to avoid a sqrt computation)
    :param p: first point
    :param q: second point
    :return: distance value
    """
    tmp = 0
    for i in range(len(p)):
        tmp += (p[i]-q[i])**2
    return tmp


def runSequential(points, k):
    """
     runSequential receives a list of tuples and an integer k.
     It comptues a 2-approximation of k points for diversity maximization
     based on matching.
    :param points: list of tuples
    :param k: integer
    :return:
    """

    n = len(points)
    if k >= n:
        return points

    result = list()
    candidates = [True for i in range(0, n)]

    # find k/2 pairs that maximize distances
    for iter in range(int(k / 2)):
        maxDist = 0.0
        maxI = 0
        maxJ = 0
        for i in range(n):
            if candidates[i]:  # Check if i is already a solution
                for j in range(i+1, n):
                    if candidates[j]:  # Check if j is already a solution
                        # use squared euclidean distance to avoid an sqrt computation!
                        d = squared_euclidean_dist(points[i], points[j])
                        if d > maxDist:
                            maxDist = d
                            maxI = i
                            maxJ = j
        result.append(points[maxI])
        result.append(points[maxJ])
        candidates[maxI] = False
        candidates[maxJ] = False

    # Add one more point if k is odd: the algorithm just start scanning
    # the input points looking for a point not in the result set.
    if k % 2 != 0:
        for i in range(n):
            if candidates[i]:
                result.append(points[i])
                break

    return result


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


def splitLine(i):
    line = i.split(",")
    return tuple(float(dim) for dim in line)


def set_kCenterMDP(k):
    def tmp(S):
        """
            receives in input a set of points S and an integer k < |S|, and returns a set C of k centers
            selected from S using the Farthest-First Traversal algorithm. It is important that kCenterMPD(S,k)
            run in O(|S|*k) time (see exercise on Slide 23 of the set of Slides on Clustering, Part 2.

        :param S: set of points
        :param k: number of points
        :return: Farthest-First Traversal algorithm result
        """

        S = list(S)  # transform intertools.chain into list
        assert k < len(S), "k < |S| is needed, but k >= |S| is found"
        random.seed(1206597)
        c0 = random.choice(S)  # correct version
        C = [c0]  # first centroid randomly selected
        S_dist = [math.inf for si in S]  # init distances of si from C
        sj_max = c0  # first element with max distance
        S_dist[0] = 0  # d(S[0], C) is 0
        for i in range(1, k):  # select other k-1 centroids with FFT algorithm
            max_distance = 0
            j_max = -1

            # Find the point sj ∈ S − C that maximizes d(sj, C)
            for j, sj in enumerate(S):
                if S_dist[j] == 0:  # sj is already in C, skip
                    continue
                # d(sj, C), C[-1] is the last added centroid
                cur_dist = min(quad_distance(sj, C[-1]), S_dist[j])
                S_dist[j] = cur_dist  # update sj distance from C
                if cur_dist > max_distance:  # check if sj is the farthest element from C
                    sj_max = sj  # save the farthest element
                    max_distance = cur_dist  # update the max distance of the element
                    j_max = j  # update the index of the farthest element

            # add the farthest element to centroid and update it's distance from C to 0
            C.append(sj_max)  # add the farthest element to C
            # set distance of the new centroid from C to 0, cause it has been added to C
            S_dist[j_max] = 0
        return C
    return tmp


def runMapReduce(pointsRDD, k, L):
    """
        implements the 4-approximation MapReduce algorithm for diversity maximization described above.

        :param pointsRDD: set of points
        :param k: number of points to extract from each partition
        :param L: number of partitions
        :return: list of tuples of extracted points
    """
    kCenterMPD = set_kCenterMDP(k)  # setup a kCenterMPD function with k as number of centroids
    start = timeit.default_timer()
    # collect action force spark execution and save points for round 2
    coreset = pointsRDD.repartition(L).mapPartitions(kCenterMPD).collect()
    stop = timeit.default_timer()
    t1 = stop-start
    # print("Runtime of Round 1 = {}".format(t1))
    start = timeit.default_timer()
    coreset = runSequential(coreset, k)  # no action required cause it's not spark
    stop = timeit.default_timer()
    t2 = stop-start
    # print("Runtime of Round 2 = {}".format(t2))
    return coreset, t1, t2


def measure(pointsSet):
    """
        computes the average distance between all pairs of points.
        :param pointsSet: set of points
        :return: average distance
    """
    mean_dist = 0
    N = len(pointsSet)
    # counter = 0
    for i in range(N):
        for j in range(i+1, N):
            mean_dist += euclidean(pointsSet[i], pointsSet[j])
            # counter += 1
    # assert counter == N*(N-1)/2, "bad counter"
    # print("computer {} distances".format(counter))
    return mean_dist/(N*(N-1)/2)


if __name__ == "__main__":

    # Check cmd line param, spark setup
    if TESTING:
        num_executor = sys.argv[4]
        test_id = sys.argv[5]
    else:
        assert len(sys.argv) == 4, "Usage: python G39HW3.py <path-to-file> <k> <L>"

    k = sys.argv[2]  # Diversity maximization parameter
    assert k.isdigit(), "K must be an integer"
    k = int(k)

    L = sys.argv[3]  # Read number of partitions
    assert L.isdigit(), "L must be an integer"
    L = int(L)

    inputPath = sys.argv[1]
    # assert os.path.isfile(inputPath), "File or folder not found"

    # spark initilization
    conf = SparkConf().setAppName('G39HW3')
    sc = SparkContext(conf=conf)
    start = timeit.default_timer()
    # cache to improve performance
    inputPointsRDD = sc.textFile(inputPath, L).map(splitLine).cache()
    num_of_points = inputPointsRDD.count()  # force spark execution
    stop = timeit.default_timer()
    init_time = stop - start
    centers, t1, t2 = runMapReduce(inputPointsRDD, k, L)
    avg_dist = measure(centers)

    if TESTING:
        print("{},{},{},{},{},{},{},{},{}".format(inputPath.split("/")[-1], k, L, num_executor, init_time, t1, t2, avg_dist, test_id))
    else:
        print("\nNumber of points = {}".format(num_of_points))
        print("k = {}".format(k))
        print("L = {}".format(L))
        print("Initialization time = {}".format(init_time))
        print("Runtime of Round 1 = {}".format(t1))
        print("Runtime of Round 2 = {}".format(t2))
        print("Average distance = {}".format(avg_dist))
