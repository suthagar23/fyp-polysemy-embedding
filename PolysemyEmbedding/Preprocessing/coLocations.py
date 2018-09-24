# Step : 2

import json
import nltk
import os
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import re
from nltk.metrics.association import QuadgramAssocMeasures
from PolysemyEmbedding.SupportModules import CommonUtils

commonUtils = CommonUtils()
config = commonUtils.getPreprocessingConfiguration()
mainConfig = commonUtils.getMainConfiguration()

basePath = mainConfig['corpusParentPath']
outputPath = commonUtils.pathJoin(basePath, config['coLocations']['coLocationsPath'])
listToken = []

NO_OF_BIGRAMS = config['coLocations']['noOfBiGrams']
NO_OF_TRIGRAMS = config['coLocations']['noOfTriGrams']
NO_OF_QUADGRAMS = config['coLocations']['noOfQuadGrams']
totalReadFilesCount = config['totalFilesToReadFromDirectory']
INPUT_FILE_PREFIX = config['inputFile']['filePrefixNotation']
INPUT_FILE_SUFFIX = config['inputFile']['fileSuffixNotation']

def runCodeForLine(corpusLines):
    for line in corpusLines:
        sentencesInLine = sent_tokenize(line)
        for sentence in sentencesInLine:
            sentence = re.sub(r'((https|http|ftp)\s*:\s*\/\s*\/\s*)?(\w{3}\s*\.).*(\.).*', '', sentence)
            sentence = ''.join(e for e in sentence if e.isalpha() or e.isspace())
            words = word_tokenize(sentence)
        for token in words:
            listToken.append(token)

#finding bigram
def findBigram (bigramsCount):
    bigram = {}
    bigramMeasures = nltk.collocations.BigramAssocMeasures()
    finder = nltk.BigramCollocationFinder.from_words(listToken)
    outputBigramFileName = "bigramOutput"
    outputBigramFile = open(commonUtils.pathJoin(outputPath + outputBigramFileName), "w+")
    listBigram = finder.nbest(bigramMeasures.pmi, bigramsCount)
    for i in listBigram:
        actualValue = ' '.join(str(e) for e in i)
        bigram[actualValue] = '_'.join(str(e) for e in i)
        # bigram = '_'.join(str(e) for e in i)
        # score = finder.score_ngram(bigramMeasures.pmi, i[0], i[1])
        # outputBigramFile.write(bigram + "\t" + str(score) + "\n")
    outputBigramFile.write(json.dumps(bigram))
    outputBigramFile.close()

# finding trigram
def findTrigram (trigramsCount):
    trigram = {}
    trigramMeasures = nltk.collocations.TrigramAssocMeasures()
    finder = nltk.TrigramCollocationFinder.from_words(listToken)
    finder.apply_freq_filter(3)
    outputTrigramFileName = "trigramOutput"
    outputTrigramFile = open(commonUtils.pathJoin(outputPath + outputTrigramFileName), "w+")
    listTrigram = finder.nbest(trigramMeasures.pmi, trigramsCount)
    for i in listTrigram:
        actualValue = ' '.join(str(e) for e in i)
        trigram[actualValue] = '_'.join(str(e) for e in i)
        # to find the pmi score
        # score = finder.score_ngram(trigramMeasures.pmi, i[0], i[1], i[2])
        # outputTrigramFile.write(trigram + "\t"+ str(score) + "\n")
    outputTrigramFile.write(json.dumps(trigram))
    outputTrigramFile.close()

#finding quadgram
def findQuadgrams (quadgramsCount):
    quadgram = {}
    quadgramMeasures = QuadgramAssocMeasures()
    finder = nltk.QuadgramCollocationFinder.from_words(listToken)
    finder.apply_freq_filter(3)
    outputQuadgramFileName = "quadgramOutput"
    outputQuadgramFile = open(commonUtils.pathJoin(outputPath + outputQuadgramFileName), "w+")
    listQuadgram = finder.nbest(quadgramMeasures.pmi, quadgramsCount)
    for i in listQuadgram:
        actualValue = ' '.join(str(e) for e in i)
        quadgram[actualValue] = '_'.join(str(e) for e in i)
        # to find the pmi score, uncomment it
        # quadgram = '_'.join(str(e) for e in i)
        # score = finder.score_ngram(quadgramMeasures.pmi, i[0], i[1], i[2], i[3])
        # outputQuadgramFile.write(quadgram + "\t" + str(score) + "\n")
    outputQuadgramFile.write(json.dumps(quadgram))
    outputQuadgramFile.close()

for i in range(0, totalReadFilesCount):
    output = {}
    counter = 0;
    totalProcess = 1;
    listToken = []
    inputFileName = INPUT_FILE_PREFIX + "{:0>2}".format(i + 1) + INPUT_FILE_SUFFIX
    print("\n\nStarted File ", i, " - ", inputFileName)
    print("-------------------------------------------------")
    inputFile = open(basePath + inputFileName, "r")
    corpusLines = inputFile.readlines()
    totalLines = len(corpusLines)
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    print(" Total Lines : ", totalLines)
    wordnet.ensure_loaded()  # first access to wn transforms it
    runCodeForLine(corpusLines)
    inputFile.close()

findQuadgrams(NO_OF_QUADGRAMS)
findTrigram(NO_OF_TRIGRAMS)
findBigram(NO_OF_BIGRAMS)

