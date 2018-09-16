
import concurrent.futures
import collections
import matplotlib
import json
import os
import time
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
from nltk.corpus import wordnet as wn
from sklearn.decomposition import PCA
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_PATH = "/media/suthagar/Data/Embedding_Sep/500_vectors/"
OUTPUT_CLUSTERS_PATH = BASE_PATH + "iter1/"
MODEL_FILENAME = "vectors.txt"
TARGET_WORDS_FILENAME="target-words.txt"

def ConcurrentClustering(lemma):

    numberOfSenses = 1
    if len(lemma)>2:
        lemmaWord = lemma[:-2]
        if lemmaWord.isalpha():
            numberOfSenses = wn.synsets(lemmaWord)
    noOfClusters = len(numberOfSenses)

    noOfClusters = 100 # Custom Clusters

    vecs = []
    words = []
    lemmaVectorMap = wordVectorMap.get(lemma)
    for key in lemmaVectorMap:
        words.append(key)
        word_vec = []
        vector_list = (lemmaVectorMap[key].split("\n"))[0].split(" ")[:-1]
        for i in (vector_list):
            word_vec.append(float(i))
        vecs.append(word_vec)
    print(lemma, "Vector Count : " + str(len(vecs)))

    if lemma in targetWordsForPlotting:
        pca = PCA(n_components=2)
        result = pca.fit_transform(vecs)
        plt.scatter(result[:, 0], result[:, 1])
        for label, x, y in zip(words, result[:, 0], result[:, 1]):
            plt.annotate(label, xy=(x, y))
        plt.savefig(OUTPUT_CLUSTERS_PATH + lemma+ " embedding.png")
        plt.show()
        plt.close()

    # cluster using kmeans
    kmeans = KMeans(n_clusters=noOfClusters, init='k-means++', max_iter=1000, n_init=1)
    kmeans.fit(vecs)

    if lemma in targetWordsForPlotting:
        # centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
        pca = PCA(n_components=2)
        result = pca.fit_transform(vecs)
        plt.title("K-means with " + str(noOfClusters) + "clusters")
        plt.scatter(result[:, 0], result[:, 1], c=labels)
        for label, x, y in zip(words, result[:, 0], result[:, 1]):
            plt.annotate(label, xy=(x, y))
        plt.savefig(OUTPUT_CLUSTERS_PATH + " " + lemma + str(noOfClusters) + "after_clustering.png")
        plt.show()
        plt.close()

    clusters = collections.defaultdict(list)
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(i)

    outputClusterName =  lemma + "-" + str(noOfClusters) + "-clusters.txt"
    # print(lemma, "Writing to cluster output file : " + outputClusterName)
    cluster_file = open(OUTPUT_CLUSTERS_PATH + outputClusterName, "w+")
    output = {}
    for cluster in range(noOfClusters):
        if cluster in clusters.keys():
            for i, sentence in enumerate(clusters[cluster]):
                # name of lemma before clustering
                old_word = words[sentence]
                # name the lemma after clustering according to the cluster id
                new_word = lemma + "_" + str(cluster)
                # append the old lemma name and the new name in a dictionary
                output[old_word] = new_word

    cluster_file.write(json.dumps(output))
    cluster_file.close()

    return outputClusterName

def createVectorMap(targetwords):
    words = {}
    with open(BASE_PATH + MODEL_FILENAME) as vectors:
        for vector in vectors:
            vec = vector.split(" ")
            parts = vec[0].split("_")
            if (len(parts) == 3):
                word = "_".join([parts[0], parts[1]])
                if (word in targetwords):
                    w = vec.pop(0)
                    if (word in words):
                        words[word][w] = " ".join(vec)
                    else:
                        dict = {}
                        dict[w] = " ".join(vec)
                        words[word] = dict
    return words


# Get the target word list
targetwordsList=set(open(BASE_PATH+TARGET_WORDS_FILENAME,"r").read().split("\n"));
targetWordsForPlotting = [""]
# Create word-vector map for the target words
wordVectorMap = createVectorMap(targetwordsList)

# Create a pool of processes
with concurrent.futures.ProcessPoolExecutor() as executor:
    startTime = time.time()
    targetLemmas = list(targetwordsList)[1:]
    # Process the list of files, but split the work across the process pool to use all CPUs!
    for lemma, result in zip(targetLemmas, executor.map(ConcurrentClustering, targetLemmas)):
        print(f"Clusters for lemma {lemma} was saved as {result}")
    endTime = time.time()
    workTime = endTime - startTime
    print("\nTime Taken : " + str(workTime) + " seconds")