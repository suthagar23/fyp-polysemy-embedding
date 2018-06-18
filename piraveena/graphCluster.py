import datetime
import json
import os

import networkx as nx
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
from datetime import datetime

path = "/home/piraveena/fyp_data/model-initial/"
target = path +"targetvectors"
modelfile = path + "model/vectors.txt"

output_image_path = "adjacency_images/"
output_cluster_path ="adjacency_clusters/"

if not os.path.exists(output_image_path):
    os.makedirs(output_image_path)

if not os.path.exists(output_cluster_path):
    os.makedirs(output_cluster_path)


def buid_model():
    print ("start building the model")
    model = KeyedVectors.load_word2vec_format(modelfile, binary=False);

    return model;

def test_similarity(model):
    print("Testing similarity greater the valuse: " + str(value))
    start = datetime.now()
    print("Process started : ", start)
    g = nx.Graph([])
    # list of vectors for lemmas
    vecs = []
    # list of lemma
    words = []
    # data = json.load(open(target)).get(lemma)
    file = target
    data = json.load(open(file)).get(lemma)
    for key in data:
        # g.add_node(key)
       for item in model.most_similar(positive= [key],topn=100):
            if item[0].startswith(lemma):
                similarity = model.similarity(key, item[0])
                if (similarity>value):
                    # print(key, item[0])
                    g.add_edge(key,item[0])
    endTime = datetime.now()
    print("\n\n Process Stopped : ", endTime)
    print("\n\n Total Duration : ", endTime - start)
    nx.draw(g)
    image = output_image_path+ str(iter)+"iter"+ str(value)+ "graph.png"
    plt.savefig(image, bbox_inches='tight')
    plt.show()
    plt.close()
    build_clusters(g)
    # no_of_senses = len(words)

def build_clusters(graph):
    print("start writing file")
    cluster_file = output_cluster_path+ str(iter)+"iter"+str(value)+ "cluster.txt"
    # print("matrix")
    list_of_clusters = nx.connected_components(graph)
    print("opening o/p file")
    cluster_file = open(cluster_file, "w+")
    output = {}
    i = 0
    for clusters in list_of_clusters:
        # print(clusters)
        new_sense = lemma + "_" + str(i)
        for oldsense in clusters:
            # append the old lemma name and the new name in a dictionary
            output[oldsense] = new_sense
        i=i+1

    cluster_file.write(json.dumps(output))
    cluster_file.close()
    print ("finished process")


iter = 0
lemma = "rock_N"
model1 = buid_model()
value = 0.8
test_similarity(model1)
