from kmeans_clustering import kmeans
from graph_clustering import graphcluster


kmeans = kmeans()
graph = graphcluster()
def kmeansClustering():
    inputfile = "/home/piraveena/fyp_data/model-50/targetvectors"
    outputpath = "pathek/"
    lemma = "book_N"
    numberOfClusters = 20
    kmeans.getInput(inputfile, outputpath, lemma, numberOfClusters)

def graph_clustering():
    inputfile = "/home/piraveena/fyp_data/model-50/model/vectors.txt"
    targetvectors = "/home/piraveena/fyp_data/model-50/targetvectors"

    outputpath = "graph/"
    lemma = "rock_N"
    value = 0.8
    graph.getInput(inputfile, outputpath, lemma, value, targetvectors)

kmeansClustering()
graph_clustering()