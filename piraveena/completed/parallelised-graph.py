import time
import json
import os

import networkx as nx

import concurrent.futures
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
from datetime import datetime

basePath = "data/"
inputfile = "/home/piraveena/fyp_data/model-50/model/vectors.txt"
targetvectors = "/home/piraveena/fyp_data/model-50/targetvectors"

outputpath = "graph/"
value = 0.8

def buid_model():
    print ("start building the model")
    model = KeyedVectors.load_word2vec_format(inputfile, binary=False);
    return model;

def test_similarity(lemma, model):
    print("Testing similarity greater the valuse: " + str(value))
    start = datetime.now()
    print("Process started at : ", start)
    graph = nx.Graph([])

    data = json.load(open(targetvectors)).get(lemma)
    for key in data:
        graph.add_node(key)
        for item in model.most_similar(positive= [key],topn=100):
            if item[0].startswith(lemma):
                similarity = model.similarity(key, item[0])
                if (similarity>value):
                    # print(key, item[0])
                    graph.add_edge(key,item[0])
    endTime = datetime.now()
    print("\n\n Process Stopped : ", endTime)
    print("\n\n Total Duration : ", endTime - start)
    nx.draw(graph)
    image =  basePath+ lemma+ ".png"
    plt.savefig(image, bbox_inches='tight')
    plt.close()
    return graph

def getOutput(graph, lemma):
    print("start writing file")
    cluster_json_file = outputpath+ lemma+ "cluster.txt" #output path
    # print("matrix")
    list_of_clusters = nx.connected_components(graph)

    print("opening o/p file")
    cluster_file = open(cluster_json_file, "w+")
    output = {}
    i = 0
    for clusters in list_of_clusters:
        # print(clusters)
        new_sense = lemma + "_" + str(i)
        for oldsense in clusters:
            # append the old lemma name and the new name in a dictionary
            output[oldsense] = new_sense
        i=i+1
    print("Number of clusters for :" + lemma + ":" + str(i))

    cluster_file.write(json.dumps(output))
    cluster_file.close()
    print ("finished process")
    return cluster_json_file

def do_process(lemma):
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    model = buid_model()
    graph = test_similarity(lemma, model)
    outputfile = getOutput(graph, lemma)
    print("output is written in ", outputfile)

def ConcurrentClustering(lemma):
    do_process (lemma)


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

