
import json
from datetime import datetime
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import multiprocessing as mp
import random
import string


basePath = "/home/suthagar/Desktop/tmp/"
outputPath = basePath + "output/"
lemmatizer = WordNetLemmatizer()

def runCodeForLine(line,index):
    outLine = {}
    outLine[index]= {}
    outLine[index]['actualLine'] = line
    # wordsInLine = []
    sentencesInLine = sent_tokenize(line)
    outLine[index]['sentTokenize'] = sentencesInLine
    outLine[index]['words'] = {}
    for sentence in sentencesInLine:
        words = (word_tokenize(sentence))
        outLine[index]['words']['wordTokenize'] = words
        # wordsInLine.append(words)
        outLine[index]['words']['info'] = {}
        cleanTokens = words[:]
        # sr = stopwords.words('english')
        wordCounter = 0
        wordOut = {}
        for token in words:
            strWordCounter = str(wordCounter)
            wordCounter += 1

            wordOut[strWordCounter] = {}
            wordOut[strWordCounter]['word'] = token
            if token in stopwords.words('english'):
                wordOut[strWordCounter]['isStopWord'] = True
            else:
                wordOut[strWordCounter]['isStopWord'] = False

                # find Synonyms,Antonyms from WordNet
                synonyms = []
                antonyms = []
                for syn in wordnet.synsets(token):
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

        outLine[index]['words']['info'] = wordOut
    return outLine[index]

def readByLines(processId, corpusLines, startLine, endLine, counter, isMultiProcessing):
    outLine = {}
    print(" Process Id : ",processId,"( From line ",startLine," to ",endLine,")")
    for line in range(startLine, endLine):
        index = str(counter)
        counter += 1
        if(corpusLines[line]!="" or corpusLines[line].replace("\n")!=""):
            outLine[index] = runCodeForLine(corpusLines[line], index)

    if(isMultiProcessing) :
        output.put(outLine)
    else:
        return outLine;


totalReadFilesCount = 2
print("Process started : ", datetime.now())
print("Total Files : ",totalReadFilesCount, end="\n")
for i in range(0, totalReadFilesCount):
    inputFileName =  "corpus-file-0" + str(i) + ".txt" # heldout-00005-of-00050
    print("\nStarted File ", i, " - ", inputFileName)
    print("-------------------------------------------------")
    outputFileName =  "corpus-file-0" + str(i) +"-output.txt"
    print("Output File ", i, " - ", outputFileName)
    inputFile = open(basePath + inputFileName, "r")
    outputFile = open(outputPath + outputFileName, "w+")

    counter=0;
    outLine = {}
    corpusLines = inputFile.readlines()
    totalLines = len(corpusLines)
    totalProcess = 4;
    allocationForProcess = int(totalLines / totalProcess)
    lastAllocation = totalLines - allocationForProcess*totalProcess
    print(" Total Lines : ",totalLines)
    print(" Total Process : ", totalProcess)
    print(" Allocation for a Process : ", allocationForProcess)
    print(" Last Allocation for a Process : ", lastAllocation)

    if (allocationForProcess > 0):
        output = mp.Queue()
        # run it using process
        processes = [mp.Process(target=readByLines, args=(processId,
                                                          corpusLines,
                                                          processId * allocationForProcess,
                                                          (processId + 1) * allocationForProcess,
                                                          processId * allocationForProcess, True)) for processId in range(totalProcess)]
        # Run processes
        for p in processes:
            p.start()
        # Exit the completed processes
        for p in processes:
            p.join()

        # Get process results from the output queue
        outLine = [output.get() for p in processes]



    # run only last one
    processId = totalProcess # single process
    outLine[totalProcess-1].update(readByLines(processId, corpusLines, processId * allocationForProcess,
                                       processId * allocationForProcess + lastAllocation,
                                       processId * allocationForProcess, False))

    outputFile.write(json.dumps(outLine))
    inputFile.close()
    outputFile.close()
    print("Output File : ", outputPath + outputFileName)
print("\n\n Process Stoped : ", datetime.now())

# Print the output
# outputFileName = outputPath + "corpus-file-00-output.txt"
# inputFile = open(outputFileName, "r")
# for line in inputFile:
#     print(line)



