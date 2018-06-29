import datetime
import json
import os

import networkx as nx
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
from datetime import datetime

class graphcluster():

    def buid_model(self, modelfile):
        print ("start building the model")
        model = KeyedVectors.load_word2vec_format(modelfile, binary=False);
        return model;

    def test_similarity(self, targetvectors, lemma, model,value ):
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
        image =  lemma+ ".png"
        plt.savefig(image, bbox_inches='tight')
        plt.close()
        return graph

    def getOutput(self,graph, lemma, outputpath):
        print("start writing file")
        cluster_json_file = outputpath+ "cluster.txt" #output path
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

        cluster_file.write(json.dumps(output))
        cluster_file.close()
        print ("finished process")
        return cluster_json_file

    def do_process(self, inputfile, outputpath, lemma, value, targetvectors):
        if not os.path.exists(outputpath):
            os.makedirs(outputpath)

        model = self.buid_model(inputfile)
        graph = self.test_similarity(targetvectors, lemma, model, value)
        outputfile = self.getOutput(graph, lemma, outputpath)
        print("output is written in ", outputfile)

    def getInput(self, inputfile, outputpath, lemma, value, targetvectors):
        self.do_process (inputfile, outputpath, lemma, value, targetvectors)

    def __init__(self):
        print("Initilized graph clustering module")

