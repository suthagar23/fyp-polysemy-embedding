import collections
import matplotlib
import json
import os


from sklearn.cluster import KMeans
from nltk.corpus import wordnet as wn
from sklearn.decomposition import PCA
matplotlib.use('Agg')
import matplotlib.pyplot as plt

basePath = "/home/piraveena/Downloads/"
output_image_path = "kmeansImages/"
output_cluster_path ="clusterPath/"

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

    for label, x, y in zip(words, result[:, 0], result[:, 1]):
        plt.annotate(label, xy=(x, y))

    plt.savefig(output_image_path + " " + lemma + "before_clustering.png")
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
    for i in range(no_cluster):
        lines = plt.plot(centroids[i, 0], centroids[i, 1], 'kx')
    plt.setp(lines, ms=15.0)
    plt.setp(lines, mew=2.0)
    plt.savefig(output_image_path + " " + lemma + str(no_cluster)+ "after_clustering.png")
    plt.show()
    plt.close()
    clusters = collections.defaultdict(list)
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(i)

    cluster_file_path = output_cluster_path + str(no_cluster) + lemma + "cluster.txt"
    print("opening o/p file")
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




lemma = "address_V"
NUMBER_OF_SENSES= 500
words, vecs = getVector(lemma)
kmeans_clustering(lemma,words, vecs, NUMBER_OF_SENSES)