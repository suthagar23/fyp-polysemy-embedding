#3 - final distribution

import csv
import json
from pprint import pprint
import tensorflow as tf
from datetime import datetime
import re
import os
import nltk
import pickle

basePath = "/media/suthagar/Data/Corpus/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/tmp/"
inputPath =  basePath + "histogram/"
# outputPath = basePath + "pre-processed-final-files/"
fDist = nltk.FreqDist()


MOST_COMMON_WORDS_COUNT = 80000          # used for embedding vocabs

# with tf.Session() as sess:
#     sess.run( tf.global_variables_initializer())
def main():
    expect_file_names = ["whole-words.csv", "top-"+ str(MOST_COMMON_WORDS_COUNT) + "-words.csv"]
    startTime = datetime.now()
    print("Process started : ", startTime)
    all_files = os.listdir(inputPath)
    fdist = nltk.FreqDist([])
    counter = 0
    for inputFileName in all_files:
        if os.path.isfile(inputPath + inputFileName) and inputFileName not in expect_file_names:
            print("\n-------------------------------------------------")
            print(" Started File", " - ", inputFileName)
            inputFile = open(inputPath + inputFileName, "rb")
            required_input = pickle.load(inputFile)
            fdist += nltk.FreqDist(required_input)
            print(" Most Frequent : " , fdist.most_common(10))
            print(" Total : " , len(fdist))
            counter+=1


    # with open(basePath + "histogram/whole-words.csv", "w+") as fp:
    #     writer = csv.writer(fp, quoting=csv.QUOTE_ALL)
    #     writer.writerows(fdist.items())

    #finding top most
    topFreq = fdist.most_common(MOST_COMMON_WORDS_COUNT)
    freq_dic = dict()
    index_dic = dict()
    count = 0
    with open(basePath + "histogram/top-" + str(MOST_COMMON_WORDS_COUNT) + "-words.csv", 'w+') as out:
        for row in topFreq:
            out.write('%s,%4d\n' % (row[0], row[1]))
            freq_dic[row[0]] = row[1]
            index_dic[row[0]] = count
            count+=1

    distrbution_file = open(basePath + "histogram/top-" + str(MOST_COMMON_WORDS_COUNT) + "-words-withCount-bin", "wb+")
    pickle.dump(freq_dic, distrbution_file)
    distrbution_file.close()

    index_file = open(basePath + "histogram/top-" + str(MOST_COMMON_WORDS_COUNT) + "-words-withIndex-bin", "wb+")
    pickle.dump(index_dic, index_file)
    index_file.close()

    endTime = datetime.now()
    print("\n Process Stopped : ", endTime)
    print("\n Duration : ", endTime - startTime)

main()