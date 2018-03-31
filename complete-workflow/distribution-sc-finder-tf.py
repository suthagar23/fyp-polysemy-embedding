# 2 - finding distributions for single corpus files

import json
from pprint import pprint
import tensorflow as tf
from datetime import datetime
import re
import os
import pickle
from printer import  Printer
import csv
import json
from pprint import pprint
import tensorflow as tf
from datetime import datetime
import re
import os
import nltk
import pickle

basePath = "/media/suthagar/Data/Corpus/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/tmp/"
inputPath =  basePath + "pre-processed-final-files/"
outputPath = basePath + "histogram/"
# listToken=[]
TRIGRAM_POS_TAG = "TGRAM"
linePrinter = Printer()



def connectData(inputFileName, directory):
    inputFile = open(directory + "/" + inputFileName, "r")
    lines = inputFile.readlines()
    data = []
    for line in lines:
        data += (tf.compat.as_str(line).split())
    return nltk.FreqDist(data)

with tf.Session() as sess:
    sess.run( tf.global_variables_initializer())
# def main():
    directories = [x[0] for x in os.walk(inputPath)][1:]
    startTime = datetime.now()
    print("Process started : ", startTime)
    for directory in directories:
        # listToken = []
        fdist = nltk.FreqDist([])
        all_files = os.listdir(directory)
        directoryBaseName = os.path.basename(directory)
        print("\n-------------------------------------------------")
        print("Started Directory", " - ", directoryBaseName)
        fileCount = 0
        totalFiles = len(all_files)-1
        for inputFileName in all_files:
            # listToken.extend(connectData(inputFileName, directory))
            fdist += connectData(inputFileName, directory)
            completedPercentage = round(fileCount/totalFiles*100,2)
            linePrinter.printNormal(completedPercentage)
            fileCount+=1

        # with open(outputPath +inputFileName+"-distribution.csv", "w+") as fp:
        #     writer = csv.writer(fp, quoting=csv.QUOTE_ALL)
        #     writer.writerows(fdist.items())

        distrbution_file = open(outputPath +inputFileName+"-distribution-bin", "wb+")
        pickle.dump(fdist, distrbution_file)
        distrbution_file.close()

    endTime = datetime.now()
    print("\nProcess Stopped : ", endTime)
    print("\nDuration : ", endTime - startTime)

# main()