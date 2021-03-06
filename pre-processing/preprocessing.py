
import json
from datetime import datetime

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

# reload(sys)
# sys.setdefaultencoding('utf8')

# basePath = "/home/suthagar/Desktop/tmp/1billion/"
basePath = "/home/piraveena/Downloads/1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/"

outputPath = basePath + "output/"
lemmatizer = WordNetLemmatizer()
linePrinter = Printer()
cachedStopWords = stopwords.words("english")
listToken = []


def runCodeForLine(line,index):
    err=0
    outLine = {}
    outLine[index]= {}
    outLine[index]['actualLine'] = line
    sentencesInLine = sent_tokenize(line)
    outLine[index]['sentTokenize'] = sentencesInLine
    outLine[index]['sentence'] = {}
    sentCounter = 0
    sentenceOut = {}
    for sentence in sentencesInLine:

        # sentence = re.sub(r'(https|http|ftp)?\s*:\s*\/\s*\/\s*(\w{3}\s*\.|\/|\?|\=|\&|\%)*\b', '', sentence)
        sentence =re.sub(r'((https|http|ftp)\s*:\s*\/\s*\/\s*)?(\w{3}\s*\.).*(\.).*', '', sentence)
        sentence = ''.join(e for e in sentence if e.isalpha() or e.isspace())

        sentenceCounter = str(sentCounter)
        sentenceOut[sentenceCounter] = {}
        sentenceOut[sentenceCounter]['actual'] = sentence
        sentenceOut[sentenceCounter]['words'] = {}
        words = word_tokenize(sentence)
        sentenceOut[sentenceCounter]['words']['wordTokenize'] = words
        sentenceOut[sentenceCounter]['words']['info'] = {}
        wordCounter = 0
        wordOut = {}

        for token in words:
            listToken.append(token)

            strWordCounter = str(wordCounter)
            wordCounter += 1

            wordOut[strWordCounter] = {}
            wordOut[strWordCounter]['word'] = token
            if token in cachedStopWords:
                wordOut[strWordCounter]['isStopWord'] = True
            else:
                wordOut[strWordCounter]['isStopWord'] = False


                #POS Tag from nltk
                wordOut[strWordCounter]['posTag']=pos_tag([token])[0][1]
                # find Synonyms,Antonyms from WordNet
                synonyms = []
                antonyms = []
                try:
                    for syn in wordnet.synsets(str(token)):
                        for lemma in syn.lemmas():
                            synonyms.append(lemma.name())
                            if lemma.antonyms():
                                antonyms.append(lemma.antonyms()[0].name())
                    synonyms = list(set(synonyms))
                    antonyms = list(set(antonyms))
                    wordOut[strWordCounter]['synonymsCount'] = len(synonyms)
                    wordOut[strWordCounter]['synonymsList'] = synonyms
                    wordOut[strWordCounter]['antonymsCount'] = len(antonyms)
                    wordOut[strWordCounter]['antonymsList'] = antonyms

                    # word Lemmatization
                    wordOut[strWordCounter]['lemmatize'] = {}
                    wordOut[strWordCounter]['lemmatize']['verb'] = lemmatizer.lemmatize(token, pos="v")
                    wordOut[strWordCounter]['lemmatize']['noun'] = lemmatizer.lemmatize(token, pos="n")
                    wordOut[strWordCounter]['lemmatize']['adjective'] = lemmatizer.lemmatize(token, pos="a")
                    wordOut[strWordCounter]['lemmatize']['adverb'] = lemmatizer.lemmatize(token, pos="r")
                except WordNetError as e:
                    # print("WordNetError on concept {}".format(token))
                    err+=1
                except AttributeError as e:
                    # print("Attribute error on concept {}: {}".format(token, e.message))
                    err += 1
                except:
                    # print("Unexpected error on concept {}: {}".format(token, sys.exc_info()[0]))
                    err += 1
            sentenceOut[sentenceCounter]['words']['info'] = wordOut

        outLine[index]['sentence'] = sentenceOut
        sentCounter+=1
    return outLine[index]

# processCompletedState=[]
def readByLines(processId, corpusLines, startLine, endLine, counter, isMultiProcessing, totalLines):
    outLine = {}
    # print(" Process Id : ",processId,"( From line ",startLine," to ",endLine,")")
    for line in range(startLine, endLine):
        if(totalLines != 1):
            completedPercentage = int((counter - startLine)/(totalLines-1)*100)
        else:
            completedPercentage = 100
        linePrinter.addValues(completedPercentage, processId)
        index = str(counter)
        counter += 1
        if(corpusLines[line]!="" or corpusLines[line].replace("\n")!=""):
            outLine[index] = runCodeForLine(corpusLines[line], index)
        output[index] = outLine[index]


# # finding trigram
# def trigram (trigramsCount):
#
#     trigramMeasures = nltk.collocations.TrigramAssocMeasures()
#     finder = nltk.TrigramCollocationFinder.from_words(listToken)
#     finder.apply_freq_filter(3)
#     listTrigram = finder.nbest(trigramMeasures.pmi, trigramsCount)
#
#     outputTrigramFileName = "trigramOutput.txt"
#     outputTrigramFile = open(outputPath + outputTrigramFileName, "w+")
#
#     for i in listTrigram:
#         trigram = '_'.join(str(e) for e in i)
#         outputTrigramFile.write(trigram + "\n")
#     outputTrigramFile.close()
#     print("Total number of words: "+ str(len(listToken)))


totalReadFilesCount = 50
output ={}
startTime = datetime.now()
print("Process started : ", startTime)
print("Total Files : ",totalReadFilesCount,"\n")
for i in range(0, totalReadFilesCount):
    output = {}
    counter = 0;
    totalProcess = 1;
    if(i<10):
        inputFileName =  "news.en.heldout-0000" + str(i) + "-of-00050" # heldout-00005-of-00050
    else:
        inputFileName = "news.en.heldout-000" + str(i) + "-of-00050"
    print("\nStarted File ", i, " - ", inputFileName)
    print("-------------------------------------------------")
    outputFileName =  "corpus-file-0" + str(i) + "- PID" + str(totalProcess) + "-output.txt"
    print("Output File ", i, " - ", outputFileName)
    inputFile = open(basePath + inputFileName, "r")
    outputFile = open(outputPath + outputFileName, "w+")

    corpusLines = inputFile.readlines()
    totalLines = len(corpusLines)
    print(" Total Process : ", totalProcess)
    if (totalLines < totalProcess):
        print(" Total needed process : ", totalLines)
        totalProcess = totalLines

    allocationForProcess = int(totalLines / totalProcess)
    lastAllocation = totalLines - allocationForProcess*totalProcess
    print(" Total Lines : ",totalLines)
    print(" Allocation for a Process : ", allocationForProcess)
    print(" Last Allocation for a Process : ", lastAllocation)

    wordnet.ensure_loaded()            # first access to wn transforms it

    if (allocationForProcess > 0):
        linePrinter.initCompletedstate(totalProcess+1)
        processes = [Thread(target=readByLines, args=(processId,
                                                          corpusLines,
                                                          processId * allocationForProcess,
                                                          (processId + 1) * allocationForProcess,
                                                          processId * allocationForProcess, True,allocationForProcess)) for processId in
                     range(totalProcess)]

        # Run processes
        for p in processes:
            p.start()
        # Exit the completed processes
        for p in processes:
            p.join()

    print("\n Running last process")
    processId = totalProcess # single process
    readByLines(processId, corpusLines, processId * allocationForProcess,
                processId * allocationForProcess + lastAllocation,
                processId * allocationForProcess, False, lastAllocation)


    outputFile.write(json.dumps(output))
    inputFile.close()
    outputFile.close()
    print("\n\nOutput File : ", outputPath + outputFileName)

endTime = datetime.now()

# trigram using nltk
no_of_trigrams = 5000
trigram(no_of_trigrams)


print("\n\n Process Stopped : ", endTime)
print("\n\n Duration : ", endTime - startTime)


# Print the output
# outputFileName = outputPath + "corpus-file-00-output.txt"
# inputFile = open(outputFileName, "r")
# for line in inputFile:
#     print(line)

