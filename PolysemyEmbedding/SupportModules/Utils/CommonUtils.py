import sys
import pickle
import json

class CommonUtils():

    mainConfiguration = {}
    preprocessingConfiguration = {}
    embeddingConfiguration = {}
    clusteringConfiguration = {}
    evaluationsConfiguration = {}

    def setConfigurations(self):
        with open("../Resources/Configs/mainConfig.json", 'r+') as f:
            self.mainConfiguration = json.load(f)
        with open("../Preprocessing/config.json", 'r+') as f:
            self.preprocessingConfiguration = json.load(f)
        with open("../Embedding/config.json", 'r+') as f:
            self.embeddingConfiguration = json.load(f)
        with open("../Clustering/config.json", 'r+') as f:
            self.clusteringConfiguration = json.load(f)
        with open("../Evaluations/config.json", 'r+') as f:
            self.evaluationsConfiguration = json.load(f)

    def getAllConfiguration(self):
        config = {
            "mainConfiguration" : self.mainConfiguration,
            "preprocessingConfiguration" : self.preprocessingConfiguration,
            "embeddingConfiguration" : self.embeddingConfiguration,
            "clusteringConfiguration" : self.clusteringConfiguration,
            "evaluationsConfiguration" : self.evaluationsConfiguration
        }
        return config

    def getMainConfiguration(self):
        return self.mainConfiguration

    def getPreprocessingConfiguration(self):
        return self.preprocessingConfiguration

    def getEmbeddingConfiguration(self):
        return self.embeddingConfiguration

    def getClusteringConfiguration(self):
        return self.clusteringConfiguration

    def getEvaluationsConfiguration(self):
        return self.evaluationsConfiguration

    def pathJoin(self, path1, path2):
        if(path1[-1] == "/") :
            return path1 + path2
        else :
            return path1 + "/" + path2
    def __init__(self):
        self.setConfigurations()


# print(CommonUtils().getPreprocessingConfiguration()['inputFile']['filePrefixNotation'])
