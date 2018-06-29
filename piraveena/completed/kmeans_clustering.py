import collections
import json
import os

from sklearn.cluster import KMeans

class kmeans():
    def getVector(self, lemma, file):
        # list of vectors for lemmas
        vecs = []
        # list of lemma
        words = []
        data = json.load(open(file)).get(lemma)
        for key in data:
            words.append(key)
            word_vec = []
            vector_list = (data[key].split("\n"))[0].split(" ")[:-1]
            for i in (vector_list):
                word_vec.append(float(i))
            vecs.append(word_vec)
        return words, vecs


    def kmeans_clustering(self, lemma, words, vecs, no_cluster):
        # cluster using kmeans
        kmeans = KMeans(n_clusters=no_cluster, init='k-means++', max_iter=1000, n_init=1)
        kmeans.fit(vecs)
        clusters = collections.defaultdict(list)
        for i, label in enumerate(kmeans.labels_):
            clusters[label].append(i)
        return clusters


    def getOutput(self, clusters,no_cluster,outputpath,words,lemma):
        cluster_json_file = outputpath +lemma + ".txt"
        print("opening o/p file")
        cluster_file = open(cluster_json_file, "w+")
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
        return cluster_json_file

    def do_process(self, lemma, inputfile, outputpath,numberOfClusters):
        if not os.path.exists(outputpath):
            os.makedirs(outputpath)

        words, vecs = self.getVector(lemma, inputfile)
        print ("started clustering")

        clustersdata = self.kmeans_clustering(lemma, words, vecs, numberOfClusters)
        print("finished clustering")

        output_file = self.getOutput(clustersdata,numberOfClusters, outputpath,words,lemma)
        print("output is written in ", output_file)

    def getInput(self, inputfile, outputpath, lemma, numberOfClusters):
        self.do_process(lemma, inputfile, outputpath, numberOfClusters)

    def __init__(self):
        print("Initilized k-means clustering module")