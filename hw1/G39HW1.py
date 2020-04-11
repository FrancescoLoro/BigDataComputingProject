from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand


def word_count_per_doc(document):
    pairs_dict = {}
    for word in document.split(' '):
        if word not in pairs_dict.keys():
            pairs_dict[word] = 1
        else:
            pairs_dict[word] += 1
    return [(key, pairs_dict[key]) for key in pairs_dict.keys()]


def word_count_1(docs):
    word_count = (docs.flatMap(word_count_per_doc)  # <-- MAP PHASE (R1)
                  .reduceByKey(lambda x, y: x + y))  # <-- REDUCE PHASE (R1)
    return word_count


def word_count_2(docs, K):
    def word_count_per_doc_random(document):
        pairs_dict = {}
        for word in document.split(' '):
            if word not in pairs_dict.keys():
                pairs_dict[word] = 1
            else:
                pairs_dict[word] += 1
        return [(rand.randint(0, K-1), (key, pairs_dict[key])) for key in pairs_dict.keys()]

    def gather_pairs(pairs):
        pairs_dict = {}
        for p in pairs[1]:
            word, occurrences = p[0], p[1]
            if word not in pairs_dict.keys():
                pairs_dict[word] = occurrences
            else:
                pairs_dict[word] += occurrences
        return [(key, pairs_dict[key]) for key in pairs_dict.keys()]

    word_count = (docs.flatMap(word_count_per_doc_random)  # <-- MAP PHASE (R1)
                  .groupByKey()                            # <-- REDUCE PHASE (R1)
                  .flatMap(gather_pairs)
                  .reduceByKey(lambda x, y: x + y))        # <-- REDUCE PHASE (R2)
    return word_count


def class_count_deterministic(docs, K):
    def class_count_per_doc_random(document):
        pairs_dict = {}
        for line in document.split("\n"):
            gamma = line.split(" ")[1]  # Takes just the class not the number
            if gamma not in pairs_dict.keys():
                pairs_dict[gamma] = 1
            else:
                pairs_dict[gamma] += 1
        return [(gamma, pairs_dict[gamma]) for gamma in pairs_dict.keys()]

    # <-- MAP PHASE (R1)
    return sorted(docs.flatMap(class_count_per_doc_random).groupByKey().mapValues(len).collect())
    # <-- REDUCE PHASE (R1)


def word_count_2_with_partition(docs):
    def gather_pairs_partitions(pairs):
        pairs_dict = {}
        for p in pairs:
            word, occurrences = p[0], p[1]
            if word not in pairs_dict.keys():
                pairs_dict[word] = occurrences
            else:
                pairs_dict[word] += occurrences
        return [(key, pairs_dict[key]) for key in pairs_dict.keys()]

    word_count = (docs.flatMap(word_count_per_doc)  # <-- MAP PHASE (R1)
                  # <-- REDUCE PHASE (R1)
                  .mapPartitions(gather_pairs_partitions)
                  .groupByKey()                              # <-- REDUCE PHASE (R2)
                  .mapValues(lambda vals: sum(vals)))

    return word_count


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
    docs.repartition(numPartitions=K)

    print("OUTPUT: \n\nVERSION WITH DETERMINISTIC PARTITIONS")

    # CLASS COUNT
    print("Number of distinct words in the documents = ",
          str(class_count_deterministic(docs, K)))

    # # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # # SETTING GLOBAL VARIABLES
    # # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # numdocs = docs.count()
    # print("Number of documents = ", numdocs)

    # # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # # STANDARD WORD COUNT with reduceByKey
    # # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # print("Number of distinct words in the documents = ",
    #       word_count_1(docs).count())

    # # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # # IMPROVED WORD COUNT with groupByKey
    # # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # print("Number of distinct words in the documents = ",
    #       word_count_2(docs, K).count())

    # # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # # IMPROVED WORD COUNT with mapPartitions
    # # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # wc = word_count_2_with_partition(docs)
    # print("Number of distinct words in the documents = ", wc.count())

    # # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # # COMPUTE AVERAGE WORD LENGTH
    # # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # average_word_len = wc.keys().map(lambda x: (x, len(x))).values().mean()
    # print("Average word length = ", average_word_len)


if __name__ == "__main__":
    main()
