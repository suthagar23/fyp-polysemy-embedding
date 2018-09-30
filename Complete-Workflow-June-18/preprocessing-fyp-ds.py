
import json
from datetime import datetime
from imp import reload
import tensorflow as tf
import concurrent.futures
import multiprocessing
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus.reader.wordnet import WordNetError
# import multiprocessing as mp
from threading import Thread

from printer import  Printer
import sys
import re
import os
# reload(sys)
# sys.setdefaultencoding('utf8')


basePath = "/media/suthagar/Data/Corpus/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/"
outputPath = "/media/suthagar/Data/Corpus/Sep29/pre-processed-files/"

lemmatizer = WordNetLemmatizer()
linePrinter = Printer()
cachedStopWords = stopwords.words("english")
listToken = []
OUTPUT_FILE_LINE_COUNT = 3000
TOTAL_FILES_FOR_READ = 99
# trigram using nltk
NO_OF_TRIGRAMS = 5000

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
    # outLine = {}
    # outLine[index]= {}
    # outLine[index]['actualLine'] = line               # Removed to reduce the output file size
    sentencesInLine = sent_tokenize(line)
    # outLine[index]['sentTokenize'] = sentencesInLine  # Removed to reduce the output file size
    # outLine['sen'] = {}
    sentCounter = 0
    sentenceOut = {}
    tmpOut = []
    for sentence in sentencesInLine:

        # sentence = re.sub(r'(https|http|ftp)?\s*:\s*\/\s*\/\s*(\w{3}\s*\.|\/|\?|\=|\&|\%)*\b', '', sentence)
        sentence =re.sub(r'((https|http|ftp)\s*:\s*\/\s*\/\s*)?(\w{3}\s*\.).*(\.).*', '', sentence)
        sentence = ''.join(e for e in sentence if e.isalpha() or e.isspace())

        sentenceCounter = str(sentCounter)
        sentenceOut[sentenceCounter] = {}
        # sentenceOut[sentenceCounter]['actual'] = sentence     # Removed to reduce the output file size
        sentenceOut[sentenceCounter]['wds'] = {}
        words = word_tokenize(sentence)

        # sentenceOut[sentenceCounter]['wds']['wdTn'] = words
        sentenceOut[sentenceCounter]['wds']['info'] = []
        wordCounter = 0
        wordOut = {}

        for token in words:
            token = token.lower()
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
                    # print("WordNetError on concept {}".format(token))
                    err+=1
                except AttributeError as e:
                    # print("Attribute error on concept {}: {}".format(token, e.message))
                    err += 1
                except:
                    # print("Unexpected error on concept {}: {}".format(token, sys.exc_info()[0]))
                    err += 1
            sentenceOut[sentenceCounter]['wds']['info'].append(wordOut[strWordCounter])
            tmpOut.append(wordOut[strWordCounter])

        # outLine = sentenceOut
        sentCounter+=1
    return tmpOut

# processCompletedState=[]
def readByLines(corpusLines, totalLines, inputFileName):
    outLine = {}
    # print(" Process Id : ",processId,"( From line ",startLine," to ",endLine,")")l
    counter = 0
    fileCounter = 0
    processDataSaved = True
    for line in range(0, totalLines):
        processDataSaved = False
        if(totalLines != 1):
            completedPercentage = round((counter)/(totalLines-1)*100,2)
        else:
            completedPercentage = 100

        # linePrinter.printNormal(completedPercentage * 8 / len(FileNames))

        index = str(counter)
        if(corpusLines[line]!="" or corpusLines[line].replace("\n")!=""):
            outLine[index] = runCodeForLine(corpusLines[line], index)
        # output[index] = outLine[index]

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
    # writeOutputFilesSchemaJSON(inputFileName, fileCounter)

def dumbOutFile(counter, fileCounter, data, inputFileName):
    outputFileName = inputFileName + "-out-" + str(fileCounter)
    outputFile = open(outputPath + inputFileName + "/" + outputFileName, "w+")
    # print(" \tDumbing : ", fileCounter*OUTPUT_FILE_LINE_COUNT, " - ", counter, outputFileName)
    outputFile.write(json.dumps(data))
    outputFile.close()

def writeOutputFilesSchemaJSON(inputFileName, fileCounter):
    outputFile = open(outputPath + inputFileName + "-output-Schemas", "w+")
    data = []
    for i in range(0, fileCounter+1):
        outputFileName = inputFileName + "/" + inputFileName + "-out-" + str(i)
        data.append(outputFileName)
    outputFile.write(str(data))
    outputFile.close()

def doWorker(inputFileName):
    inputFile = open(basePath + inputFileName, "r")
    corpusLines = inputFile.readlines()
    totalLines = len(corpusLines)
    print(" Started File - ", inputFileName, " | Total Lines : ", totalLines)
    if not os.path.exists(outputPath + inputFileName):
        os.makedirs(outputPath + inputFileName)
    readByLines(corpusLines, totalLines, inputFileName)
    inputFile.close()
    return inputFileName

startTime = datetime.now()
print("Process started : ", startTime)
print("Total Files : ", TOTAL_FILES_FOR_READ)
FileNames = []
for i in range(0, TOTAL_FILES_FOR_READ):
    listToken = []
    if (i < 9):
        FileNames.append("news.en-0000" + str(i + 1) + "-of-00100")
    else:
        FileNames.append("news.en-000" + str(i + 1) + "-of-00100")


wordnet.ensure_loaded()
with concurrent.futures.ProcessPoolExecutor() as executor:
    for lemma, result in zip(FileNames, executor.map(doWorker, FileNames)):
        print(f"Finished for input file {lemma} was saved inside the {result}")

endTime = datetime.now()
print("\n\n Process Stopped : ", endTime)
print("\n\n Duration : ", endTime - startTime)
