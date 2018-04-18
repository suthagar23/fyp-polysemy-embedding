import collections
import matplotlib
import json
from datetime import datetime
import os

import numpy as np
import sys
from gensim.models import Word2Vec
from sklearn.cluster import KMeans, DBSCAN
from nltk.corpus import wordnet as wn
from sklearn.decomposition import PCA
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

basePath = "/home/piraveena/Downloads/1-billion-word-language-modeling-benchmark-r13output/sample_final/news.en-00055-of-00100/"

output_modelPath = "output_model/"
input_text_for_model_path = "text_For_Modeling/"
output_image_path = "images/"
output_cluster_path = "clusters/"

def no_of_senses(lemma):
    if lemma.isalpha():
        senses = wn.synsets(lemma)
    return len(senses)

# for the first time, we need to load all files into a single file
def writefile():
    file = input_text_for_model_path + "result_text_address_before_iteration_0"
    f = open(file, "w")
    for i in range(1, 10000):
        inputFile = basePath + "news.en-00055-of-00100-out-" + str(i) + "-output.txt"
        # print(" Started File", " - ", inputFile)
        lines = open(inputFile).read()
        f.write(lines)

    f.close()
    return file

#building model
def build_model(lemma, fileName, iter):
    sentences = []
    j = 0
    print("Process started : ")
    print("\n-------------------------------------------------")
    lines = open(fileName).read()
    st = ""
    if (iter == 1):
        # for the first iteration, we need to label the lemma according to their occurence in the file.
        for line in lines.split("\n"):
            words = line.split(" ")
            for num in range(0, len(words)):
                if (words[num].startswith(lemma)):
                    j += 1
                    # print(words[num])
                    words[num] = words[num] + "_" + str(j)
            st += (" ".join(words))
            st += "\n"
            sentences.append(words)

    else:
        # load the names of lemma

        file = output_cluster_path + str(iter-1) + lemma  + "cluster.txt"
        data = json.load(open(file))
        for line in lines.split("\n"):
            words = line.split(" ")
            for num in range(0, len(words)):
                if (words[num].startswith(lemma)):
                    # print(words[num])
                    words[num] = data[words[num]]
            st += (" ".join(words))
            st += "\n"
            sentences.append(words)
        # print ("j"+ str(j))
    # create a new file for embedding and write new lines with modified lemma names
    f = open(input_text_for_model_path+"result_text_address_before_iteration_"+str(iter), "w")
    f.write(st)
    f.close()
    # building new model
    model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)
    model.save(output_modelPath +str(iter)+'model.bin')

def load_model(iter):
    model = Word2Vec.load(output_modelPath +str(iter)+'model.bin')
    return model

def getVector(lemma, model, iter):
    # list of vectors for lemmas
    vecs = []
    # list of lemma
    words = []
    for word in model.wv.vocab:
        if word.startswith(lemma):
            words.append(word)
            vecs.append(model[word])
    pca = PCA(n_components=2)

    result = pca.fit_transform(vecs)
    plt.scatter(result[:, 0], result[:, 1])
    # print (words)
    for label, x, y in zip(words, result[:, 0], result[:, 1]):
        plt.annotate(label,xy=(x,y))
    # print(words)
    plt.savefig(output_image_path + str(iter)+lemma+"before_clustering.png")
    plt.show()
    plt.close()
    return vecs,words

def kmeans_clustering(lemma, vecs, words, no_cluster,iter):
    # cluster using kmeans
    kmeans = KMeans(n_clusters=no_cluster, init='k-means++', max_iter=1000, n_init=1)
    kmeans.fit(vecs)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    pca = PCA(n_components=2)
    result = pca.fit_transform(vecs)

    plt.title("K-means with " +str(no_cluster) + "clusters")
    plt.scatter(result[:, 0], result[:, 1], c=labels)
    for i in range(no_cluster):
        lines = plt.plot(centroids[i, 0], centroids[i, 1], 'kx')
    plt.setp(lines, ms=15.0)
    plt.setp(lines, mew=2.0)
    plt.savefig(output_image_path +str(iter)+lemma+"after_clustering.png")
    plt.show()
    plt.close()
    clusters = collections.defaultdict(list)
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(i)

    cluster_file_path = output_cluster_path + str(iter) + lemma  + "cluster.txt"
    print("opening o/p file")
    cluster_file = open(cluster_file_path, "w+")
    output = {}
    for cluster in range(no_cluster):
        if cluster in clusters.keys():
            for i, sentence in enumerate(clusters[cluster]):
                # name of lemma before clustering
                old_word = words[sentence]
                # name the lemma after clustering according to the cluster id
                new_word = words[sentence].split("_")[0] + "_" + str(cluster)
                # append the old lemma name and the new name in a dictionary
                output[old_word] = new_word

    cluster_file.write(json.dumps(output))
    cluster_file.close()
    return cluster_file_path

def dbscan_clustering(lemma, vecs,words,iter,eps):
    X = StandardScaler().fit_transform(vecs)
    db = DBSCAN(eps=eps, min_samples=1).fit(X)
    labels = db.labels_
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print("Number of clusters in iteration "+ str(iter) + ": "+ str(n_clusters) )
    # print("Estimated {:d} clusters".format(n_clusters), file=sys.stderr)

    output = {}
    # output = [{'word': w, 'label': np.asscalar(l), 'isCore': i in core_indices} for i, (l, w) in enumerate(zip(labels, words))]
    for i, (l, w) in enumerate(zip(labels, words)):
        old_word = w
        clusterNo = np.asscalar(l)
        new_word = old_word.split("_")[0]+"_"+str(clusterNo)
        output[old_word] = new_word

    # print(json.dumps(output))
    cluster_file_path = output_cluster_path + str(iter)+lemma + "cluster.txt"
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
        plt.annotate(label,xy=(x,y))
    if (len(unique_labels)>0):
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]
            class_member_mask = (labels == k)
            # xy = X[class_member_mask & core_samples_mask]
            # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=6)
            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o',markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters)
    plt.savefig(output_image_path + str(iter)+ lemma +"after_clustering.png")
    plt.show()
    plt.close()
    return cluster_file_path

def main(lemma, no_of_iter):
    global input_text_for_model_path
    global output_image_path
    global output_cluster_path
    global output_modelPath

    input_text_for_model_path = input_text_for_model_path + lemma +"/"
    output_image_path = output_image_path  + lemma +"/"
    output_cluster_path = output_cluster_path  + lemma +"/"
    output_modelPath = output_modelPath  + lemma +"/"

    if not os.path.exists(input_text_for_model_path):
        os.makedirs(input_text_for_model_path)

    if not os.path.exists(output_cluster_path):
        os.makedirs(output_cluster_path)

    if not os.path.exists(output_image_path):
        os.makedirs(output_image_path)

    if not os.path.exists(output_modelPath):
        os.makedirs(output_modelPath)

    eps = 13
    for iter in range (1,no_of_iter):
        if (iter ==1):
            # clustering using kmeans
            nclusters = no_of_senses(lemma)
            print(nclusters)
            print("start reading for first time")
            writefile()

        inputfileForModel = input_text_for_model_path+"result_text_address_before_iteration_"+str(iter-1)

        print ("start building model in iteration: " +str(iter))
        build_model(lemma, inputfileForModel,iter)
        print ("finished building model in iteration: " +str(iter))
        model = load_model(iter)
        print ("finished loading model in iteration: " +str(iter))
        vectors,wordsList = getVector(lemma, model,iter)
        print ("start clustering in iteration: " +str(iter))
        start_time = datetime.now()
        if(iter == 1):
            kmeans_clustering(lemma, vectors, wordsList, nclusters, 1)

        else:
            # dbscan clustering
            dbscan_clustering(lemma, vectors,wordsList, iter, eps-2)
        ending_time = datetime.now()
        time_taken_for_clustering = ending_time - start_time
        print ("finished clustering in iteration: " +str(iter) + " in "+ str(time_taken_for_clustering))
        # #---------------------

    print ("completed")

if __name__ == '__main__':
    lemma = "book"
    no_of_iter = 5
    main(lemma, no_of_iter)