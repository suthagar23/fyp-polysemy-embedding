# Former Name : preprocessing-fyp-tf.py
# Step : 1

import json
from datetime import datetime
import tensorflow as tf
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus.reader.wordnet import WordNetError
from PolysemyEmbedding.SupportModules import LinePrinter
from PolysemyEmbedding.SupportModules import CommonUtils
import re
import os


commonUtils = CommonUtils()
config =commonUtils.getPreprocessingConfiguration()
mainConfig = commonUtils.getMainConfiguration()


basePath = mainConfig['corpusParentPath']
corpusPath = commonUtils.pathJoin(basePath, config['corpusFileDirectory'])
outputPath = commonUtils.pathJoin(basePath, config['preprocessFileDirectory'])
lemmatizer = WordNetLemmatizer()
linePrinter = LinePrinter()
cachedStopWords = stopwords.words("english")
listToken = []
OUTPUT_FILE_LINE_COUNT = config['numberOfLinesPerFileToWrite']
TOTAL_FILES_FOR_READ = config['totalFilesToReadFromDirectory']
INPUT_FILE_PREFIX = config['inputFile']['filePrefixNotation']
INPUT_FILE_SUFFIX = config['inputFile']['fileSuffixNotation']
OUTPUT_FILE_SUFFIX = config['inputFile']['filePrefixNotation']

# map nltk pos tag results to wordnet pos tag arguments
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

def runCodeForLine(line,index):
    err=0
    sentencesInLine = sent_tokenize(line)
    sentCounter = 0
    sentenceOut = {}
    tmpOut = []
    for sentence in sentencesInLine:
        sentence =re.sub(r'((https|http|ftp)\s*:\s*\/\s*\/\s*)?(\w{3}\s*\.).*(\.).*', '', sentence)
        sentence = ''.join(e for e in sentence if e.isalpha() or e.isspace())

        sentenceCounter = str(sentCounter)
        sentenceOut[sentenceCounter] = {}
        sentenceOut[sentenceCounter]['wds'] = {}
        words = word_tokenize(sentence)
        sentenceOut[sentenceCounter]['wds']['info'] = []
        wordCounter = 0
        wordOut = {}

        for token in words:
            listToken.append(token)

            strWordCounter = str(wordCounter)
            wordCounter += 1
            wordOut[strWordCounter] = {}
            wordOut[strWordCounter]['word'] = token           # Removed to reduce the output file size
            if token in cachedStopWords:
                wordOut[strWordCounter]['isST'] = True
            else:
                wordOut[strWordCounter]['isST'] = False
                #POS Tag from nltk
                nltk_pos_tag = pos_tag([token])[0][1]
                wordOut[strWordCounter]['posT']= nltk_pos_tag
                try:
                    wordOut[strWordCounter]['lem'] = {}
                    wordOut[strWordCounter]['lem']['v'] = lemmatizer.lemmatize(token, pos="v")
                    wordOut[strWordCounter]['lem']['n'] = lemmatizer.lemmatize(token, pos="n")
                    wordOut[strWordCounter]['lem']['a'] = lemmatizer.lemmatize(token, pos="a")
                    wordOut[strWordCounter]['lem']['r'] = lemmatizer.lemmatize(token, pos="r")
                except WordNetError as e:
                    print("WordNetError on concept {}".format(token))
                    err+=1
                except AttributeError as e:
                    print("Attribute error on concept {}: {}".format(token, e.message))
                    err += 1
                except:
                    print("Unexpected error on concept {}: {}".format(token))
                    err += 1
            sentenceOut[sentenceCounter]['wds']['info'].append(wordOut[strWordCounter])
            tmpOut.append(wordOut[strWordCounter])
        sentCounter+=1
    return tmpOut

def readByLines(corpusLines, totalLines, inputFileName):
    outLine = {}
    counter = 0
    fileCounter = 0
    processDataSaved = True
    for line in range(0, totalLines):
        processDataSaved = False
        if(totalLines != 1):
            completedPercentage = round((counter)/(totalLines-1)*100,2)
        else:
            completedPercentage = 100
        linePrinter.printNormal(completedPercentage)
        index = str(counter)
        if(corpusLines[line]!="" or corpusLines[line].replace("\n")!=""):
            outLine[index] = runCodeForLine(corpusLines[line], index)
        if(counter%OUTPUT_FILE_LINE_COUNT == 0 and counter!=0):
            dumbOutFile(counter,fileCounter, outLine, inputFileName)
            processDataSaved = True
            outLine = {}
            fileCounter += 1
        counter += 1

    # For last batch data
    if not processDataSaved:
        dumbOutFile(counter, fileCounter, outLine, inputFileName)
    # Finally write a file with the created file names
    writeOutputFilesSchemaJSON(inputFileName, fileCounter)

def dumbOutFile(counter, fileCounter, data, inputFileName):
    outputFileName = inputFileName + "-out-" + str(fileCounter)
    outputFile = open(commonUtils.pathJoin(commonUtils.pathJoin(outputPath, inputFileName),outputFileName), "w+")
    outputFile.write(json.dumps(data))
    outputFile.close()

def writeOutputFilesSchemaJSON(inputFileName, fileCounter):
    outputFile = open(commonUtils.pathJoin(outputPath , inputFileName + OUTPUT_FILE_SUFFIX), "w+")
    data = []
    for i in range(0, fileCounter+1):
        outputFileName = commonUtils.pathJoin(inputFileName,inputFileName + "-out-" + str(i))
        data.append(outputFileName)
    outputFile.write(str(data))
    outputFile.close()

with tf.Session() as sess:
    sess.run( tf.global_variables_initializer())
    startTime = datetime.now()
    print("Process started : ", startTime)
    print("Total Files : ",TOTAL_FILES_FOR_READ)
    for i in range(0, TOTAL_FILES_FOR_READ):
        listToken = []
        inputFileName = INPUT_FILE_PREFIX + "{:0>2}".format(i + 1) + INPUT_FILE_SUFFIX
        print("\n\nStarted File ", str(i+1), " - ", inputFileName)
        print("-------------------------------------------------")
        inputFile = open(commonUtils.pathJoin(corpusPath, inputFileName), "r")
        corpusLines = inputFile.readlines()
        totalLines = len(corpusLines)
        print(" Total Lines : ", totalLines)
        wordnet.ensure_loaded()            # first access to wn transforms it
        if not os.path.exists(commonUtils.pathJoin(outputPath, inputFileName)):
            os.makedirs(commonUtils.pathJoin(outputPath, inputFileName))
        readByLines(corpusLines, totalLines, inputFileName)
        inputFile.close()

    endTime = datetime.now()
    print("\n\n Process Stopped : ", endTime)
    print("\n\n Duration : ", endTime - startTime)



