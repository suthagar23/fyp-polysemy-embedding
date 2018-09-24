import matplotlib
matplotlib.use('Agg')
from similarity_clustering.similarity_cluster import config_similariy_cluster,start_similarity_clustering
from kmeans_clustering.kmeans_cluster import config_kmeans_cluster,start_kmeans
from target_vectors.target_vectors import config_target_vectors,get_needed_vectors

def embedding():
    iteration="iteration-1"
    inputFile="/home/piraveena/Documents/Semi8/fyp-polysemy-embedding/data/histogram/text8"

def configure_target_vectors():
    basePath = "data/"
    inputFile = "model/vectors.txt"
    targetWordsFile = "top-500-polywords.txt"
    # Create a text file names as target-words.txt which has the list of words(with pos tag) that need to be clustered.
    outputfile = "top-500-vectors.txt"
    config_target_vectors(basePath,inputFile,targetWordsFile,outputfile)
    get_needed_vectors()


def configure_similarity_cluster():

    basePath = "/home/piraveena/fyp/EndToEnd/data/"
    inputfile = "/home/piraveena/fyp_data/model-50/model/vectors.txt"
    targetvectors = "/home/piraveena/fyp_data/model-50/targetvectors"
    outputpath = "graph/"
    value = 0.8
    config_similariy_cluster(basePath,inputfile,targetvectors,outputpath,value)
    start_similarity_clustering()


def configure_kmeans_cluster():
    basePath = "data/"
    outputImagePath = basePath + "images/"
    outputClusterPath = basePath + "textfiles/"
    senses = 4
    targetvectors ="/home/piraveena/fyp_data/model-50/targetvectors"
    config_kmeans_cluster(basePath,outputImagePath,outputClusterPath,senses,targetvectors)
    start_kmeans()



if __name__ == '__main__':
    # config_similariy_cluster()
    configure_kmeans_cluster()