import concurrent.futures
import time

import collections
import matplotlib
import json
import os

from sklearn.cluster import KMeans
from nltk.corpus import wordnet as wn
from sklearn.decomposition import PCA
matplotlib.use('Agg')
import matplotlib.pyplot as plt

basePath = "data/"
output_image_path = basePath+ "images/"
output_cluster_path =basePath+ "textfiles/"
NUMBER_OF_SENSES = 4


if not os.path.exists(output_image_path):
    os.makedirs(output_image_path)

if not os.path.exists(output_cluster_path):
    os.makedirs(output_cluster_path)

def no_of_senses(lemma):
    if lemma.isalpha():
        senses = wn.synsets(lemma)
    return len(senses)


def getVector(lemma):
    # list of vectors for lemmas
    vecs = []
    # list of lemma
    words = []
    file = basePath + "targetvectors"
    data = json.load(open(file)).get(lemma)
    # print(len(data))

    for key in data:
        words.append(key)
        word_vec = []
        vector_list = (data[key].split("\n"))[0].split(" ")[:-1]
        for i in (vector_list):
            word_vec.append(float(i))
        vecs.append(word_vec)

    pca = PCA(n_components=2)

    result = pca.fit_transform(vecs)
    plt.scatter(result[:, 0], result[:, 1])
    # plt.figure(figsize=(20, 5))
    for label, x, y in zip(words, result[:, 0], result[:, 1]):
        plt.annotate(label, xy=(x, y))

    # plt.savefig(output_image_path + " " + lemma + "before_clustering.png")
    plt.savefig(output_image_path + "Before clustering: " + lemma+ ": occurence: "+ str(len(data)) +" .png")
    plt.show()
    plt.close()
    return words,vecs

def kmeans_clustering(lemma,words,vecs,no_cluster):
    # cluster using kmeans
    kmeans = KMeans(n_clusters=no_cluster, init='k-means++', max_iter=1000, n_init=1)
    kmeans.fit(vecs)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    pca = PCA(n_components=2)
    result = pca.fit_transform(vecs)

    plt.title("K-means with " + str(no_cluster) + "clusters")
    plt.scatter(result[:, 0], result[:, 1], c=labels)
    for label, x, y in zip(words, result[:, 0], result[:, 1]):
        plt.annotate(label, xy=(x, y))
    plt.savefig( output_image_path + " " + "After clustering: "+ lemma +": clusters: " + str(no_cluster)+ ".png")
    plt.show()
    plt.close()
    clusters = collections.defaultdict(list)
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(i)

    cluster_file_path = output_cluster_path + lemma + ": clusters: "+ str(no_cluster) +".txt"
    print("writing into cluster text file" + lemma)
    cluster_file = open(cluster_file_path, "w+")
    output = {}
    for cluster in range(no_cluster):
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
    return cluster_file_path

def ConcurrentClustering(lemma):
    lemma = lemma
    words, vecs = getVector(lemma)
    kmeans_clustering(lemma, words, vecs, NUMBER_OF_SENSES)
    print ("finished clustering: " + lemma)

# Create a pool of processes
with concurrent.futures.ProcessPoolExecutor() as executor:
    startTime = time.time()
    #define the list of target words
    with open(basePath+"target-words.txt") as f:
        lemmas = f.read().splitlines()
    # Process the list of files, but split the work across the process pool to use all CPUs!
    for lemma, result in zip(lemmas, executor.map(ConcurrentClustering, lemmas)):
        print("Clusters for lemma: "+ lemma)
    endTime = time.time()
    workTime = endTime - startTime
    print("\nTime Taken : " + str(workTime) + " seconds")