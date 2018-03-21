import json
from pprint import pprint
import tensorflow as tf
from datetime import datetime
import re
import os

basePath = "/media/suthagar/Data/Corpus/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/delete/"
inputPath =  basePath + "pre-processed-files/"
outputPath = basePath + "pre-processed-final-files/"
listToken=[]
TRIGRAM_POS_TAG = "TGRAM"

def connectData(inputFileName, directory):
    data = json.load(open(directory + "/" + inputFileName))
    for senIndex,wordInfo in data.items():
        wordIndex = 0
        while wordIndex < len(wordInfo):
            listToken.append(wordInfo[wordIndex]['word'])
            wordIndex+=1
    f=open(inputPath + "total-word-list-output.txt","w")
    f.write(str(listToken))
    f.close()


with tf.Session() as sess:
    sess.run( tf.global_variables_initializer())
    directories = [x[0] for x in os.walk(inputPath)][1:]
    startTime = datetime.now()
    print("Process started : ", startTime)
    for directory in directories:
        all_files = os.listdir(directory)
        if len(all_files)>0 :
            if not os.path.exists(outputPath + os.path.basename(directory)):
                os.makedirs(outputPath + os.path.basename(directory))
        for inputFileName in all_files:
            print("\n\nStarted File", " - ", inputFileName)
            print("-------------------------------------------------")
            connectData(inputFileName, directory)

    endTime = datetime.now()
