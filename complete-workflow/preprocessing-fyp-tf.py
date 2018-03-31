
import json
from datetime import datetime
from imp import reload
import tensorflow as tf
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

# basePath = "/home/suthagar/Desktop/tmp/1billion/"
baseTopPath = "/media/suthagar/Data/Corpus/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/"
basePath = baseTopPath + "tmp/"
outputPath = basePath + "pre-processed-files/"
lemmatizer = WordNetLemmatizer()
linePrinter = Printer()
cachedStopWords = stopwords.words("english")
listToken = []
OUTPUT_FILE_LINE_COUNT = 10000
START_FILE_ID = 16  #set default as 0
TOTAL_FILES_FOR_READ = 34
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
                    # find Synonyms,Antonyms from WordNet
                    # synonyms = []
                    # antonyms = []
                    # for syn in wordnet.synsets(str(token)):
                    #     for lemma in syn.lemmas():
                    #         synonyms.append(lemma.name())
                    #         if lemma.antonyms():
                    #             antonyms.append(lemma.antonyms()[0].name())
                    # synonyms = list(set(synonyms))
                    # antonyms = list(set(antonyms))
                    # wordOut[strWordCounter]['synC'] = len(synonyms)
                    # wordOut[strWordCounter]['synL'] = synonyms
                    # wordOut[strWordCounter]['antC'] = len(antonyms)
                    # wordOut[strWordCounter]['antL'] = antonyms

                    # word Lemmatization
                    # wordnet_postag = get_wordnet_pos(nltk_pos_tag)
                    # if wordnet_postag == 'x':
                    #     lemmatized_word = lemmatizer.lemmatize(token)
                    # else:
                    #     lemmatized_word = lemmatizer.lemmatize(token, pos=wordnet_postag)
                    # wordOut[strWordCounter]['lemWrd'] = lemmatized_word
                    # print(lemmatized_word, pos_tag([token]))

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
        linePrinter.printNormal(completedPercentage)
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

# finding trigram
def findTrigram (trigramsCount, inputFileName):

    trigramMeasures = nltk.collocations.TrigramAssocMeasures()
    finder = nltk.TrigramCollocationFinder.from_words(listToken)
    finder.apply_freq_filter(3)
    listTrigram = finder.nbest(trigramMeasures.pmi, trigramsCount)

    outputTrigramFileName = inputFileName + "-trigram-output"
    outputTrigramFile = open(outputPath + outputTrigramFileName, "w+")
    trigram = {}
    for i in listTrigram:
        actualValue = ' '.join(str(e) for e in i)
        trigram[actualValue] = '_'.join(str(e) for e in i)
    outputTrigramFile.write(json.dumps(trigram))
    outputTrigramFile.close()
    # print("Total number of words: "+ str(len(listToken)))

with tf.Session() as sess:
    sess.run( tf.global_variables_initializer())
    startTime = datetime.now()
    print("Process started : ", startTime)
    print("Total Files : ",TOTAL_FILES_FOR_READ)
    for i in range(START_FILE_ID, TOTAL_FILES_FOR_READ):
        listToken = []
        if(i<9):
            inputFileName =  "news.en-0000" + str(i+1) + "-of-00100" # heldout-00005-of-00050
        else:
            inputFileName = "news.en-000" + str(i+1) + "-of-00100"
        print("\n\nStarted File ", i, " - ", inputFileName)
        print("-------------------------------------------------")
        startFileTime = datetime.now()
        inputFile = open(baseTopPath + inputFileName, "r")
        corpusLines = inputFile.readlines()
        totalLines = len(corpusLines)
        print(" Total Lines : ", totalLines)
        wordnet.ensure_loaded()            # first access to wn transforms it
        if not os.path.exists(outputPath + inputFileName):
            os.makedirs(outputPath + inputFileName)
        readByLines(corpusLines, totalLines, inputFileName)
        inputFile.close()
        findTrigram(NO_OF_TRIGRAMS, inputFileName)
        endFileTime = datetime.now()
        print(" File Duration : ", endFileTime - startFileTime)
    endTime = datetime.now()
    print("\n\n Process Stopped : ", endTime)
    print("\n\n Total Duration : ", endTime - startTime)


# Print the output
# outputFileName = outputPath + "corpus-file-00-output.txt"
# inputFile = open(outputFileName, "r")
# for line in inputFile:
#     print(line)

