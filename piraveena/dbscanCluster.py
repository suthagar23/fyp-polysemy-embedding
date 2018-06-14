import os
import matplotlib
import numpy as np
import json
from sklearn.cluster import DBSCAN
from nltk.corpus import wordnet as wn
from sklearn.decomposition import PCA
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

basePath = "/home/piraveena/Documents/target_vectors_data/"
output_image_path = "dbclusterImages/"
output_cluster_path ="dbscanCluster/"

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
    file = basePath + "targetvectors-50"

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

    # plt.savefig(output_image_path + " " + lemma + "before_clustering.png")
    plt.savefig(output_image_path + "embedding.png")
    plt.show()
    plt.close()
    return words,vecs


def dbscan(vecs, words,eps):
    min = 2
    X = StandardScaler().fit_transform(vecs)
    db = DBSCAN(eps=eps, min_samples= min).fit(X)
    labels = db.labels_
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print("Number of clusters in iteration " + str(iter) + ": " + str(n_clusters))
    # print("Estimated {:d} clusters".format(n_clusters), file=sys.stderr)

    output = {}
    # output = [{'word': w, 'label': np.asscalar(l), 'isCore': i in core_indices} for i, (l, w) in enumerate(zip(labels, words))]
    for i, (l, w) in enumerate(zip(labels, words)):
        old_word = w
        clusterNo = np.asscalar(l)
        new_word = lemma + "_" + str(clusterNo)
        # append the old lemma name and the new name in a dictionary
        output[old_word] = new_word

    # output = {}
    # for cluster in range(no_cluster):
    #     if cluster in clusters.keys():
    #         for i, sentence in enumerate(clusters[cluster]):
    #             # name of lemma before clustering
    #             old_word = words[sentence]
    #             # name the lemma after clustering according to the cluster id
    #             new_word = lemma + "_" + str(cluster)
    #             # append the old lemma name and the new name in a dictionary
    #             output[old_word] = new_word

    # cluster_file.write(json.dumps(output))
    # cluster_file.close()
    # return cluster_file_path
    # print(json.dumps(output))
    cluster_file_path = output_cluster_path + str(eps)+"_"+str(min)+ "cluster.txt"
    cluster_file = open(cluster_file_path, "w+")
    cluster_file.write(json.dumps(output))
    cluster_file.close()
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    pca = PCA(n_components=3)
    result = pca.fit_transform(vecs)
    plt.scatter(result[:, 0], result[:, 1], c=labels)
    for label, x, y in zip(words, result[:, 0], result[:, 1]):
        plt.annotate(label, xy=(x, y))
    if (len(unique_labels) > 0):
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]
            class_member_mask = (labels == k)
            # xy = X[class_member_mask & core_samples_mask]
            # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=6)
            # xy = X[class_member_mask & ~core_samples_mask]
            # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters)
    plt.savefig(output_image_path + str(eps)+"_"+str(min)+" after_clustering.png")
    plt.show()
    plt.close()
    return cluster_file_path


lemma = "address_N"
words, vecs = getVector(lemma)
dbscan(vecs, words, 15.5)

