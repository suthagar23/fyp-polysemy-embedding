import json
from pprint import pprint
import tensorflow as tf
from datetime import datetime
import re
import os
from printer import Printer

basePath = "/media/suthagar/Data/Corpus/Sep29/"
inputPath = basePath + "pre-processed-final-files/"
outPath = basePath

linePrinter = Printer()

def connectData(inputFileName, directory):
    data = open(directory + "/" + inputFileName).readlines()
    f = open(outPath + "preprocessed-corpus.txt", "a")
    f.write("\n".join(data) + "\n")
    f.close()



directories = [x[0] for x in os.walk(inputPath)][1:]
startTime = datetime.now()
print("Process started : ", startTime)
fileCount = 0
for directory in directories:
    print(" Started Directory", " - ", os.path.basename(directory))
    all_files = os.listdir(directory)
    for inputFileName in all_files:
        # print(" Started File", " - ", inputFileName)
        connectData(inputFileName, directory)
        fileCount += 1

data = open(outPath + "preprocessed-corpus.txt").readlines()
print(len(data))
print( fileCount)

endTime = datetime.now()
print("\nProcess Stopped : ", endTime)
print("\nDuration : ", endTime - startTime)
