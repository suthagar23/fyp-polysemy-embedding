# 1 - create Final Corpus


import json
from pprint import pprint
import tensorflow as tf
from datetime import datetime
import re
import os
from printer import  Printer
from nltk.corpus import stopwords

# basePath = "/media/suthagar/Data/Corpus/pre-processed/"
basePath = "/media/suthagar/Data/Corpus/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/tmp/"
inputPath =  basePath + "pre-processed-files/"
outputPath = basePath + "pre-processed-final-files/"
trigramData = []
TRIGRAM_POS_TAG = "TGRAM"
linePrinter = Printer()
cachedStopWords = stopwords.words("english")

topWordsSample = ["address","book","government","company","say","make","take","get","come","give"]
topWordsSampleCount = [0,0,0,0,0,0,0,0,0,0]

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

def notInStopWords(word):
    token = word.lower()
    if token in cachedStopWords:
        # print(word)
        return False
    else:
        return True

def returnFinalWordForPolysemy(word, tag):
    word = word.lower()
    tag = tag.upper()
    if word in topWordsSample:
        index = topWordsSample.index(word)
        finalWord = word + "_" + tag + "_" + str(topWordsSampleCount[index])
        topWordsSampleCount[index] += 1
        return  finalWord
    else:
        return word + "_" + tag

def connectData(inputFileName, directory):
    data = json.load(open(directory + "/" + inputFileName))
    output=""
    for senIndex,wordInfo in data.items():
        wordIndex = 0
        while wordIndex < len(wordInfo):
            trigramFound = False
            nextIsTrigram = False
            if (wordIndex == 0 or wordIndex != len(wordInfo)-1) and len(wordInfo)>2:

                if wordIndex == 0:
                    trigramMatch = wordInfo[wordIndex]['word'] + " " + wordInfo[wordIndex + 1]['word'] + " " + \
                                   wordInfo[wordIndex + 2]['word']
                else:
                    trigramMatch = wordInfo[wordIndex-1]['word'] + " " + wordInfo[wordIndex]['word'] + " " + wordInfo[wordIndex+1]['word']
                if trigramMatch in trigramData:
                    # print(wordIndex, "Trigram found : " + trigramData[trigramMatch])
                    wordIndex+=1
                    if wordIndex != 0:
                        output += " " + trigramData[trigramMatch].lower() + "_" + TRIGRAM_POS_TAG
                    else:
                        output += trigramData[trigramMatch].lower() + "_" + TRIGRAM_POS_TAG
                    trigramFound = True
                else:
                    try:
                        trigramMatch = wordInfo[wordIndex]['word'] + " " + wordInfo[wordIndex+1]['word'] + " " + \
                                   wordInfo[wordIndex + 2]['word']
                        if trigramMatch in trigramData:
                            # to skip the current word appending, next iteration will replace the trigram
                            nextIsTrigram = True
                    except IndexError:
                        pass


            if not trigramFound and not nextIsTrigram:
                if not wordInfo[wordIndex]['isST']:  # ignores stop words
                    if notInStopWords(wordInfo[wordIndex]['word']):
                        if 'lemWrd' in wordInfo[wordIndex]:
                            # for new files
                            # word = wordInfo[wordIndex]['lemWrd'].lower() + "_" + wordInfo[wordIndex]['posT']
                            word = returnFinalWordForPolysemy(wordInfo[wordIndex]['lemWrd'], wordInfo[wordIndex]['posT'])
                        else:
                            # for old files
                            pos_tag_wn = get_wordnet_pos(wordInfo[wordIndex]['posT'])
                            if pos_tag_wn is not 'x':
                                # word = wordInfo[wordIndex]['lem'][pos_tag_wn].lower() + "_" + pos_tag_wn.upper()
                                word = returnFinalWordForPolysemy(wordInfo[wordIndex]['lem'][pos_tag_wn], pos_tag_wn)
                            else:
                                # word = wordInfo[wordIndex]['word'].lower() + "_" + wordInfo[wordIndex]['posT']
                                word = returnFinalWordForPolysemy(wordInfo[wordIndex]['word'], wordInfo[wordIndex]['posT'])
                        if wordIndex != 0:
                            output += " " + word
                        else:
                            output += word

            wordIndex+=1
        output += ".\n"


    f=open(outputPath + os.path.basename(directory) + "/" + inputFileName + "-output.txt","w")
    f.write(output)
    f.close()

with tf.Session() as sess:
    sess.run( tf.global_variables_initializer())
    directories = [x[0] for x in os.walk(inputPath)][1:]
    startTime = datetime.now()
    print("Process started : ", startTime)
    for directory in directories:
        print("\n-------------------------------------------------")
        print(" Started Directory", " - ", os.path.basename(directory))
        all_files = os.listdir(directory)
        if len(all_files)>0 :
            if not os.path.exists(outputPath + os.path.basename(directory)):
                os.makedirs(outputPath + os.path.basename(directory))
        fileCount = 0
        for inputFileName in all_files:
            print(" Started File", " - ", inputFileName)
            # completedPercentage = round(fileCount/(len(all_files)-1)*100,2)
            # linePrinter.printNormal(completedPercentage)
            # load the trigram file
            trigramData = json.load(open(inputPath + os.path.basename(directory) + "-trigram-output"))
            connectData(inputFileName, directory)
            fileCount+=1

    endTime = datetime.now()
    print("\nProcess Stopped : ", endTime)
    print("\nDuration : ", endTime - startTime)




