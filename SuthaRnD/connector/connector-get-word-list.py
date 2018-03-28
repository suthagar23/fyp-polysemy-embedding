import json
from pprint import pprint
import tensorflow as tf
from datetime import datetime
import re
import os
import pickle
from printer import  Printer

basePath = "/media/suthagar/Data/Corpus/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/tmp/"
inputPath =  basePath + "pre-processed-files/"
outputPath = basePath + "histogram/"
# listToken=[]
TRIGRAM_POS_TAG = "TGRAM"
linePrinter = Printer()

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'
    elif treebank_tag.startswith('V'):
        return 'v'
    elif treebank_tag.startswith('N'):
        return 'n'
    elif treebank_tag.startswith('R'):
        return 'r'
    else:
        # all other tags get mapped to x
        return 'x'

def connectData(inputFileName, directory):
    currentListToken = []
    data = json.load(open(directory + "/" + inputFileName))
    for senIndex, wordInfo in data.items():
        wordIndex = 0
        while wordIndex < len(wordInfo):
            if not wordInfo[wordIndex]['isST']:  # ignores stop words
                pos_tag_wn = get_wordnet_pos(wordInfo[wordIndex]['posT'])
                if pos_tag_wn is not 'x':
                    word = wordInfo[wordIndex]['lem'][pos_tag_wn]
                else:
                    word = wordInfo[wordIndex]['word']
                currentListToken.append(word)
            wordIndex += 1
    return currentListToken


with tf.Session() as sess:
    sess.run( tf.global_variables_initializer())
# def main():
    directories = [x[0] for x in os.walk(inputPath)][1:]
    startTime = datetime.now()
    print("Process started : ", startTime)
    for directory in directories:
        listToken = []
        all_files = os.listdir(directory)
        directoryBaseName = os.path.basename(directory)
        print("\n-------------------------------------------------")
        print("Started Directory", " - ", directoryBaseName)
        fileCount = 0
        totalFiles = len(all_files)-1
        for inputFileName in all_files:
            listToken.extend(connectData(inputFileName, directory))
            completedPercentage = round(fileCount/totalFiles*100,2)
            linePrinter.printNormal(completedPercentage)
            fileCount+=1

        f = open(outputPath + "text-format/" + directoryBaseName + "-all-words-txt", "w")
        f.write(str(listToken))
        f.close()

        lemmas_file = open(outputPath + directoryBaseName + "-all-words-bin", "wb+")
        pickle.dump(listToken, lemmas_file)
        lemmas_file.close()
        print("\n Total Words(with our Stop words) : ", len(listToken))
        print(" Words Dump Name", " - ", directoryBaseName + "-all-words-bin")


    endTime = datetime.now()
    print("\nProcess Stopped : ", endTime)
    print("\nDuration : ", endTime - startTime)

# main()