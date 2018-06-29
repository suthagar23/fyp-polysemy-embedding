from kmeans_clustering import kmeans


kmeans = kmeans()
def kmeansClustering():
    inputfile = "/home/piraveena/fyp_data/model-50/targetvectors"
    outputpath = "pathek/"
    lemma = "book_N"
    numberOfClusters = 20
    kmeans.getinput(inputfile, outputpath, lemma, numberOfClusters)


kmeansClustering()