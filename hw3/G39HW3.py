# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# METHOD runSequential
# Sequential 2-approximation for diversity maximization based on matching
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# Import Packages
import os
import sys

from pyspark import SparkConf, SparkContext


def squared_euclidean_dist(p, q):
    """
    Compute the squared euclidean distance (to avoid a sqrt computation)
    :param p:
    :param q:
    :return:
    """
    tmp = 0
    for i in range(0, len(p) - 1):
        tmp += (p[i] - q[i]) ** 2
    return tmp


# runSequential receives a list of tuples and an integer k.
# It comptues a 2-approximation of k points for diversity maximization
# based on matching.
def runSequential(points, k):
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
                for j in range(n):
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


def runMapReduce(pointsRDD, k, L):
    """
        implements the 4-approximation MapReduce algorithm for diversity maximization described above.

        :param pointsRDD: set of points
        :param k: number of points to extract from each partition
        :param L: number of partitions
        :return: list of tuples of extracted points
    """
    return 0


def measure(pointsSet):
    """
        computes the average distance between all pairs of points.
        :param pointsSet: set of points
        :return: average distance
    """
    return 0 


if __name__ == "__main__":

    # Check cmd line param, spark setup
    assert len(sys.argv) == 3, "Usage: python G39HW3.py <path-to-file> <k> <L>"

    # spark initilization
    conf = SparkConf().setAppName('G39HW3')
    sc = SparkContext(conf=conf)

    k = sys.argv[2]  # Diversity maximization parameter
    assert k.isdigit(), "K must be an integer"
    k = int(k)

    L = sys.argv[3]  # Read number of partitions
    assert L.isdigit(), "L must be an integer"
    L = int(L)

    inputPath = sys.argv[1]
    assert os.path.isfile(inputPath), "File or folder not found"
    inputPoints = sc.textFile(inputPath).map(f).repartition(L).cache()  # Read input tuples

    print("\nNumber of points = {}".format(len(inputPoints)))
    print("\nk = {}".format(k))
    print("\nL = {}".format(L))
    print("\nInitialization time = {}".format(k))
