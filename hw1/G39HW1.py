from pyspark import SparkContext, SparkConf
import sys
import os


def class_count_per_doc(document):
    pairs_dict = {}
    for line in document.split("\n"):
        gamma = line.split(" ")[1]  # Takes just the class not the number
        if gamma not in pairs_dict.keys():
            pairs_dict[gamma] = 1
        else:
            pairs_dict[gamma] += 1
    return [(gamma, pairs_dict[gamma]) for gamma in pairs_dict.keys()]

def count_in_a_partition(idx, iterator):
  count = 0
  for _ in iterator:
    count += 1
  return idx, count

def count_max_partition(docs):
    max_partition_size = docs.mapPartitionsWithIndex(count_in_a_partition).collect()
    del max_partition_size[0::2] #Remove index keep count
    return max(max_partition_size)

def class_count_deterministic(docs):
    return sorted(docs.flatMap(class_count_per_doc).groupByKey().mapValues(len).collect())# <-- MAP PHASE (R1) # <-- REDUCE PHASE (R1)


def class_count_with_spark_partition(docs):
    def gather_pairs_partitions(pairs):
        pairs_dict = {}
        for p in pairs:
            word, occurrences = p[0], p[1]
            if word not in pairs_dict.keys():
                pairs_dict[word] = occurrences
            else:
                pairs_dict[word] += occurrences
        return [(key, pairs_dict[key]) for key in pairs_dict.keys()]

    word_count = (docs.flatMap(class_count_per_doc)  # <-- MAP PHASE (R1)  # <-- REDUCE PHASE (R1)
                  .mapPartitions(gather_pairs_partitions)
                  .groupByKey()                              # <-- REDUCE PHASE (R2)
                  .mapValues(lambda vals: sum(vals)))

    #Find most frequent class
    max_count = 0
    max_class = 0
    for c, count in word_count.collect():
        if count > max_count:
            max_count = count
            max_class = c

    return (max_class, max_count), count_max_partition(docs)


def main():
    # Check cmd line param, spark setup
    assert len(sys.argv) == 3, "Usage: python TemplateHW1.py <K> <file_name>"
    conf = SparkConf().setAppName('G39HW1').setMaster("local[*]")
    sc = SparkContext(conf=conf)

    # Read number of partitions
    K = sys.argv[1]
    assert K.isdigit(), "K must be an integer"
    K = int(K)

    # Read input file and subdivide it into K random partitions
    data_path = sys.argv[2]
    assert os.path.isfile(data_path), "File or folder not found"
    docs = sc.textFile(data_path, minPartitions=K).cache()
    docs = docs.repartition(numPartitions=K)

    print("INPUT: \n\n** K =", K)
    print("\n**DATASET:", data_path)
    print("\nOUTPUT:\n")

    # CLASS COUNT
    print("VERSION WITH DETERMINISTIC PARTITIONS")
    print("Output pairs = ",
          str(class_count_deterministic(docs)))

    # CLASS COUNT WITH SPARK PARTITIONS
    print("VERSION WITH SPARK PARTITIONS")
    max_count, max_partition_size  = class_count_with_spark_partition(docs)
    print("Most frequent class = ", str(max_count))
    print("Max partition size = ", max_partition_size)

if __name__ == "__main__":
    main()
