# Former Name : connector-corpus-creator.py
# Step : 3

import json
import tensorflow as tf
from datetime import datetime
import os
from PolysemyEmbedding.SupportModules import LinePrinter
from PolysemyEmbedding.SupportModules import CommonUtils

commonUtils = CommonUtils()
config = commonUtils.getPreprocessingConfiguration()
mainConfig = commonUtils.getMainConfiguration()

basePath = mainConfig['corpusParentPath']
inputPath = commonUtils.pathJoin(basePath, config['preprocessFileDirectory'])
outputPath = commonUtils.pathJoin(basePath, config['embeddingFilesDirectory'])
colocationsPath = commonUtils.pathJoin(basePath, config['coLocations']['coLocationsPath'])
QUADGRAM_POS_TAG = config['coLocations']['quadGramPosTag']
TRIGRAM_POS_TAG = config['coLocations']['triGramPosTag']
BIGRAM_POS_TAG = config['coLocations']['biGramPosTag']
linePrinter = LinePrinter()


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
    data = json.load(open(commonUtils.pathJoin(directory, inputFileName)))
    output = ""
    for senIndex, wordInfo in data.items():
        wordIndex = 0
        while wordIndex < len(wordInfo):
            ngramFound = False
            if (wordIndex == 0 or wordIndex != len(wordInfo) - 1) and len(wordInfo) > 2:
                if len(wordInfo) > 3 and wordIndex < len(wordInfo) - 2:
                    if wordIndex == 0:
                        quadgramMatch = wordInfo[wordIndex]['word'] + " " + wordInfo[wordIndex + 1]['word'] + " " + \
                                        wordInfo[wordIndex + 2]['word'] + " " + wordInfo[wordIndex + 3]['word']

                    else:
                        quadgramMatch = wordInfo[wordIndex - 1]['word'] + " " + wordInfo[wordIndex]['word'] + " " + \
                                        wordInfo[wordIndex + 1]['word'] + " " + wordInfo[wordIndex + 2]['word']

                    if quadgramMatch in quadgramData:
                        if wordIndex != 0:
                            output += " " + str(quadgramData[quadgramMatch]).lower() + "_" + QUADGRAM_POS_TAG
                            wordIndex += 4
                        else:
                            output += str(quadgramData[quadgramMatch]).lower() + "_" + QUADGRAM_POS_TAG
                            wordIndex += 3
                        ngramFound = True
                elif len(wordInfo) > 2:
                    if wordIndex == 0:
                        trigramMatch = wordInfo[wordIndex]['word'] + " " + wordInfo[wordIndex + 1]['word'] + " " + \
                                       wordInfo[wordIndex + 2]['word']
                    else:
                        trigramMatch = wordInfo[wordIndex - 1]['word'] + " " + wordInfo[wordIndex]['word'] + " " + \
                                       wordInfo[wordIndex + 1]['word']
                    if trigramMatch in trigramData:
                        if wordIndex != 0:
                            output += " " + str(trigramData[trigramMatch]).lower() + "_" + TRIGRAM_POS_TAG
                            wordIndex += 3
                        else:
                            output += str(trigramData[trigramMatch]).lower() + "_" + TRIGRAM_POS_TAG
                            wordIndex += 2
                        ngramFound = True
                elif len(wordInfo) > 1:
                    bigramMatch = wordInfo[wordIndex]['word'] + " " + wordInfo[wordIndex + 1]['word']

                    if bigramMatch in bigramData:
                        wordIndex += 2
                        output += " " + str(bigramData[bigramMatch]) + "_" + BIGRAM_POS_TAG
                    ngramFound = True

            if not ngramFound:
                if not wordInfo[wordIndex]['isST']:  # ignores stop words
                    if 'lemWrd' in wordInfo[wordIndex]:
                        # for new files
                        if (wordInfo[wordIndex]['posT'].lower() == "v"):
                            word = str(wordInfo[wordIndex]['lemWrd']).lower() + "_" + wordInfo[wordIndex]['posT']
                        else:
                            word = str(wordInfo[wordIndex]['word']).lower() + "_" + wordInfo[wordIndex]['posT']
                    else:
                        # for old files
                        if (wordInfo[wordIndex]['posT'].lower() == "v"):
                            pos_tag_wn = get_wordnet_pos(wordInfo[wordIndex]['posT'])
                            if pos_tag_wn is not 'x':
                                word = str(wordInfo[wordIndex]['lem'][pos_tag_wn]).lower() + "_" + pos_tag_wn.upper()
                            else:
                                word = str(wordInfo[wordIndex]['word']).lower() + "_" + wordInfo[wordIndex]['posT']
                        else:
                            word = str(wordInfo[wordIndex]['word']).lower() + "_" + wordInfo[wordIndex]['posT']

                    if wordIndex != 0:
                        output += " " + word
                    else:
                        output += word
            wordIndex += 1
        if(output.strip()!=''):
            output += ".\n"

    f = open(commonUtils.pathJoin(commonUtils.pathJoin(outputPath, os.path.basename(directory)),
                                  inputFileName + "-output.txt"), "w")
    f.write(output)
    f.close()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    directories = [x[0] for x in os.walk(inputPath)][1:]
    startTime = datetime.now()
    print("Process started : ", startTime)
    for directory in directories:
        print("\n-------------------------------------------------")
        print(" Started Directory", " - ", os.path.basename(directory))
        all_files = os.listdir(directory)
        if len(all_files) > 0:
            if not os.path.exists(commonUtils.pathJoin(outputPath, os.path.basename(directory))):
                os.makedirs(commonUtils.pathJoin(outputPath, os.path.basename(directory)))
        fileCount = 0
        bigramData = json.load(open(commonUtils.pathJoin(colocationsPath , config['coLocations']['biGramFileName'])))
        trigramData = json.load(open(commonUtils.pathJoin(colocationsPath , config['coLocations']['triGramFileName'])))
        quadgramData = json.load(open(commonUtils.pathJoin(colocationsPath , config['coLocations']['quadGramFileName'])))

        for inputFileName in all_files:
            print(" Started File", " - ", inputFileName)
            connectData(inputFileName, directory)
            fileCount += 1

    endTime = datetime.now()
    print("\nProcess Stopped : ", endTime)
    print("\nDuration : ", endTime - startTime)
