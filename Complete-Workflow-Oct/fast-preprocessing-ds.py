
import concurrent.futures
import json
import re
import time
from datetime import datetime

from nltk import pos_tag_sents
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from printer import Printer
import multiprocessing

basePath = "/home/ubuntu/work/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/"
outputPath = "/home/ubuntu/work/pre-processed-files/"
colocationsPath = "/home/ubuntu/work/colocations/"

lemmatizer = WordNetLemmatizer()
linePrinter = Printer()
cachedStopWords = stopwords.words("english")
OUTPUT_FILE_LINE_COUNT = 3000
TOTAL_FILES_FOR_READ = 99

QUADGRAM_POS_TAG = "QGRAM"
TRIGRAM_POS_TAG = "TGRAM"
BIGRAM_POS_TAG = "BGRAM"

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

def makeCorpus(data, lock):
    output=""
    outputWP = ""
    for senIndex,wordInfo in data.items():
        wordIndex = 0
        while wordIndex < len(wordInfo):
            ngramFound = False
            if (wordIndex == 0 or wordIndex != len(wordInfo)-1) and len(wordInfo)>2:
                if len(wordInfo)>3 and wordIndex< len(wordInfo)-2:
                    if wordIndex == 0:
                        quadgramMatch = wordInfo[wordIndex][0] + " " + wordInfo[wordIndex + 1][0] + " " + \
                                       wordInfo[wordIndex + 2][0] + " " + wordInfo[wordIndex + 3][0]

                    else:
                        quadgramMatch = wordInfo[wordIndex - 1][0] + " " + wordInfo[wordIndex][0] + " " + \
                                        wordInfo[wordIndex + 1][0] + " " + wordInfo[wordIndex + 2][0]

                    if quadgramMatch in quadgramData:
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
                        trigramMatch = wordInfo[wordIndex][0] + " " + wordInfo[wordIndex + 1][0] + " " + \
                                       wordInfo[wordIndex + 2][0]
                    else:
                        trigramMatch = wordInfo[wordIndex-1][0] + " " + wordInfo[wordIndex][0] + " " + wordInfo[wordIndex+1][0]
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
                    bigramMatch = wordInfo[wordIndex][0] + " " + wordInfo[wordIndex + 1][0]


                    if bigramMatch in bigramData:
                        wordIndex += 2
                        output += " " + str(bigramData[bigramMatch]) + "_" + BIGRAM_POS_TAG
                        outputWP += " " + str(bigramData[bigramMatch])
                    ngramFound = True

            if not ngramFound:
                if wordInfo[wordIndex][0] not in cachedStopWords:  # ignores stop words
                    pos_tag_wn = get_wordnet_pos(wordInfo[wordIndex][1])
                    if pos_tag_wn is not "x":
                        try:
                            lemmatizedWord = lemmatizer.lemmatize(wordInfo[wordIndex][0], pos=pos_tag_wn)
                        except:
                            lemmatizedWord = wordInfo[wordIndex][0]
                        word = lemmatizedWord.lower() + "_" + wordInfo[wordIndex][1]
                        wordWP = lemmatizedWord.lower()
                    else:
                        word = str(wordInfo[wordIndex][0]).lower() + "_" + wordInfo[wordIndex][1]
                        wordWP = str(wordInfo[wordIndex][0]).lower()

                    if wordIndex != 0:
                        output += " " + word
                        outputWP += " " + wordWP
                    else:
                        output += word
                        outputWP += wordWP
            wordIndex+=1
        output += ".\n"
        outputWP += ".\n"
    with lock:
        f = open(outputPath + "preprocessed-corpus-withPOS-" + fileUniqueId + ".txt", "a")
        f.write(output)
        f.close()
        f = open(outputPath + "preprocessed-corpus-withoutPOS-" + fileUniqueId + ".txt", "a")
        f.write(outputWP)
        f.close()


def runCodeForLine(line):
    sentencesInLine = sent_tokenize(line.lower())
    tokens = []
    for sentence in sentencesInLine:
        sentence =re.sub(r'((https|http|ftp)\s*:\s*\/\s*\/\s*)?(\w{3}\s*\.).*(\.).*', '', sentence)
        sentence = ''.join(e for e in sentence if e.isalpha() or e.isspace())
        words = word_tokenize(sentence)
        tokens.append(words)
    posTaggedWords = pos_tag_sents(tokens)
    return posTaggedWords[0]

def readByLines(corpusLines, totalLines, inputFileName, lock):
    outLine = {}
    counter = 0
    for line in range(0, totalLines):
        index = str(counter)
        if(corpusLines[line]!="" or corpusLines[line].replace("\n")!=""):
            outLine[index] = runCodeForLine(corpusLines[line])
        counter += 1

        if counter%2000 == 0:
            makeCorpus(outLine, lock)
            outLine = {}
            print(" Writing to corpus file from : " + inputFileName + " [" + str(counter) + " / " + str(totalLines) +"]")

def dumbOutFile(counter, fileCounter, data, inputFileName):
    outputFileName = inputFileName + "-out-" + str(fileCounter)
    outputFile = open(outputPath + inputFileName + "/" + outputFileName, "w+")
    # print(" \tDumbing : ", fileCounter*OUTPUT_FILE_LINE_COUNT, " - ", counter, outputFileName)
    outputFile.write(json.dumps(data))
    outputFile.close()

def doWorker(inputFileName, lock):
    print(lock)
    inputFile = open(basePath + inputFileName, "r")
    corpusLines = inputFile.readlines()
    totalLines = len(corpusLines)
    print(" Started File - ", inputFileName, " | Total Lines : ", totalLines)
    # if not os.path.exists(outputPath + inputFileName):
    #     os.makedirs(outputPath + inputFileName)
    readByLines(corpusLines, totalLines, inputFileName, lock)
    inputFile.close()
    return inputFileName

startTime = datetime.now()
print("Process started : ", startTime)
print("Total Files : ", TOTAL_FILES_FOR_READ)
bigramData = json.load(open(colocationsPath + "bigramOutput"))
trigramData = json.load(open(colocationsPath + "trigramOutput"))
quadgramData = json.load(open(colocationsPath + "quadgramOutput"))
fileUniqueId = str(int(time.time()))

FileNames = []
for i in range(0, TOTAL_FILES_FOR_READ):
    if (i < 9):
        FileNames.append("news.en-0000" + str(i + 1) + "-of-00100")
    else:
        FileNames.append("news.en-000" + str(i + 1) + "-of-00100")


wordnet.ensure_loaded()
pool = concurrent.futures.ProcessPoolExecutor()
m = multiprocessing.Manager()
lock = m.Lock()
futures = [pool.submit(doWorker, FileName, lock) for FileName in FileNames]
for future in futures:
    print(f"Finished and saved inside the {future.result()}")

endTime = datetime.now()
print("\n\n Process Stopped : ", endTime)
print("\n\n Duration : ", endTime - startTime)
