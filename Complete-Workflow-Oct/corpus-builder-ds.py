import json
from pprint import pprint
import tensorflow as tf
from datetime import datetime
import time
import re
import os
from printer import  Printer
import concurrent.futures
import multiprocessing

basePath = "/media/suthagar/Data/Corpus/Sep29/"
inputPath =  basePath + "pre-processed-files/"
outputPath = basePath
colocationsPath = basePath + "colocations/"

QUADGRAM_POS_TAG = "QGRAM"
TRIGRAM_POS_TAG = "TGRAM"
BIGRAM_POS_TAG = "BGRAM"
linePrinter = Printer()

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'
    elif treebank_tag.startswith('V'):
        return 'v'
    elif treebank_tag.startswith('N') and treebank_tag.endswith('S'):
        return 'n'
    elif treebank_tag.startswith('R'):
        return 'r'
    else:
        # all other tags get mapped to x
        return 'x'

def doWorker(lemma):
    data = json.load(open(lemma))
    output=""
    outputWP = ""
    for senIndex,wordInfo in data.items():
        wordIndex = 0
        while wordIndex < len(wordInfo):
            ngramFound = False
            if (wordIndex == 0 or wordIndex != len(wordInfo)-1) and len(wordInfo)>2:
                if len(wordInfo)>3 and wordIndex< len(wordInfo)-2:
                    if wordIndex == 0:
                        quadgramMatch = wordInfo[wordIndex]['word'] + " " + wordInfo[wordIndex + 1]['word'] + " " + \
                                       wordInfo[wordIndex + 2]['word'] + " " + wordInfo[wordIndex + 3]['word']

                    else:
                        quadgramMatch = wordInfo[wordIndex - 1]['word'] + " " + wordInfo[wordIndex]['word'] + " " + \
                                        wordInfo[wordIndex + 1]['word'] + " " + wordInfo[wordIndex + 2]['word']

                    if quadgramMatch in quadgramData:
                        # print(wordIndex, "Trigram found : " + trigramData[trigramMatch])

                        if wordIndex != 0:
                            output += " " + str(quadgramData[quadgramMatch]).lower() + "_" + QUADGRAM_POS_TAG
                            outputWP += " " + str(quadgramData[quadgramMatch]).lower()
                            wordIndex += 4
                        else:
                            output += str(quadgramData[quadgramMatch]).lower() + "_" + QUADGRAM_POS_TAG
                            outputWP += str(quadgramData[quadgramMatch]).lower()
                            wordIndex += 3
                        ngramFound = True
                elif len(wordInfo)>2:
                    if wordIndex == 0:
                        trigramMatch = wordInfo[wordIndex]['word'] + " " + wordInfo[wordIndex + 1]['word'] + " " + \
                                       wordInfo[wordIndex + 2]['word']
                    else:
                        trigramMatch = wordInfo[wordIndex-1]['word'] + " " + wordInfo[wordIndex]['word'] + " " + wordInfo[wordIndex+1]['word']
                    if trigramMatch in trigramData:
                        if wordIndex != 0:
                            output += " " + str(trigramData[trigramMatch]).lower() + "_" + TRIGRAM_POS_TAG
                            outputWP += " " + str(trigramData[trigramMatch]).lower()
                            wordIndex += 3
                        else:
                            output += str(trigramData[trigramMatch]).lower() + "_" + TRIGRAM_POS_TAG
                            outputWP += str(trigramData[trigramMatch]).lower()
                            wordIndex += 2
                        ngramFound = True
                elif len(wordInfo) > 1:
                    bigramMatch = wordInfo[wordIndex]['word'] + " " + wordInfo[wordIndex + 1]['word']


                    if bigramMatch in bigramData:
                        wordIndex += 2
                        output += " " + str(bigramData[bigramMatch]) + "_" + BIGRAM_POS_TAG
                        outputWP += " " + str(bigramData[bigramMatch])
                    ngramFound = True

            if not ngramFound:
                if not wordInfo[wordIndex]['isST']:  # ignores stop words
                    pos_tag_wn = get_wordnet_pos(wordInfo[wordIndex]['posT'])
                    if 'lemWrd' in wordInfo[wordIndex]:
                        # for new files
                        if pos_tag_wn is not "x":
                            word = str(wordInfo[wordIndex]['lemWrd']).lower() + "_" + wordInfo[wordIndex]['posT']
                            wordWP = str(wordInfo[wordIndex]['lemWrd']).lower()
                        else:
                            word = str(wordInfo[wordIndex]['word']).lower() + "_" + wordInfo[wordIndex]['posT']
                            wordWP = str(wordInfo[wordIndex]['word']).lower()
                    else:
                        # for old files
                        if pos_tag_wn is not 'x':
                            word = str(wordInfo[wordIndex]['lem'][pos_tag_wn]).lower() + "_" + pos_tag_wn.upper()
                            wordWP = str(wordInfo[wordIndex]['lem'][pos_tag_wn]).lower()
                        else:
                            word = str(wordInfo[wordIndex]['word']).lower() + "_" + wordInfo[wordIndex]['posT']
                            wordWP = str(wordInfo[wordIndex]['word']).lower()

                    if wordIndex != 0:
                        output += " " + word
                        outputWP += " " + wordWP
                    else:
                        output += word
                        outputWP += wordWP
            wordIndex+=1
        output += ".\n"
        outputWP += ".\n"

    f = open(outputPath + "preprocessed-corpus-withPOS-" + fileUniqueId + ".txt", "a")
    f.write(output)
    f.close()
    f = open(outputPath + "preprocessed-corpus-withoutPOS-" + fileUniqueId + ".txt", "a")
    f.write(outputWP)
    f.close()

bigramData = json.load(open(colocationsPath + "bigramOutput"))
trigramData = json.load(open(colocationsPath + "trigramOutput"))
quadgramData = json.load(open(colocationsPath + "quadgramOutput"))
fileUniqueId = str(int(time.time()))

directories = [x[0] for x in os.walk(inputPath)][1:]
startTime = datetime.now()
print("Corpus merger : Started")
# print("Process started : ", startTime)
FileNameList = []
for directory in directories:
    all_files = os.listdir(directory)
    for inputFileName in all_files:
        FileNameList.append(directory + "/" + inputFileName)


fileCount = 0
totalFiles = len(FileNameList)
print(" Total files to be merged : " , totalFiles)
with concurrent.futures.ProcessPoolExecutor() as executor:
    for lemma, result in zip(FileNameList, executor.map(doWorker, FileNameList)):
        fileCount += 1
        completedPercentage = int(fileCount/totalFiles * 100)
        linePrinter.printNormal(completedPercentage)
print("\n Corpus with POS Tag is saved to : " + "preprocessed-corpus-withPOS-" + fileUniqueId + ".txt")
print(" Corpus without POS Tag is saved to : " + "preprocessed-corpus-withoutPOS-" + fileUniqueId + ".txt")
endTime = datetime.now()
# print("\nProcess Stopped : ", endTime)
print("\nDuration : ", endTime - startTime)
