#4 - build dataset

import csv
import json
from pprint import pprint
import tensorflow as tf
from datetime import datetime
import re
import os
import nltk
import pickle
import numpy as np
import collections
import math
import os
import random

basePath = "/media/suthagar/Data/Corpus/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/tmp/"
histogramInputPath =  basePath + "histogram/"
inputPath = basePath + "pre-processed-final-files/"
MOST_COMMON_WORDS_COUNT = 80000

def loadFinalVocabsFile():
    inputFile = open(histogramInputPath + "top-"+ str(MOST_COMMON_WORDS_COUNT)+"-words-withIndex-bin", "rb")
    required_input = pickle.load(inputFile)
    required_input["UNK"] = -1
    count = 0
    for key in required_input:
        print(key, required_input[key])
        count+=1
        if(count>15):
            break
    return required_input

vocabsData_dictionary  = loadFinalVocabsFile()
reverse_dictionary = dict(zip(vocabsData_dictionary.values(), vocabsData_dictionary.keys()))
data = []

def build_dataset(words):
  unk_count = 0
  wordData = []
  for word in words:
    index = vocabsData_dictionary.get(word, 0)
    if index == 0:  # dictionary['UNK']
      unk_count += 1
    wordData.append(index)
  vocabsData_dictionary['UNK'] += unk_count
  return wordData

# Read the data into a list of strings.
def read_data(inputFileName):
  inputFile = open(inputFileName, "r")
  lines = inputFile.readlines()
  data = []
  for line in lines:
    data+=(tf.compat.as_str(line).split())
  return data

all_dir = os.listdir(inputPath)
for inputFileDir in all_dir:
  if(os.path.isdir(inputPath + inputFileDir)):
    all_files = os.listdir(inputPath + inputFileDir)
    print(len(all_files))
    for inputFileName in all_files:
      actualFileName = inputPath + inputFileDir + "/" + inputFileName
      if os.path.isfile(actualFileName):
        print("Processing file : ", inputFileName)
        data.extend(build_dataset(read_data(actualFileName)))
        print('Data size', len(data))

print('vocabsData_dictionary size', len(vocabsData_dictionary))
print('reverse_dictionary size', len(reverse_dictionary))
print('Data size', len(data))

print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])


data_file = open(basePath + "embedding-files/data-bin", "wb+")
pickle.dump(data, data_file)
data_file.close()

reverse_dictionary_file = open(basePath + "embedding-files/reverse_dictionary-bin", "wb+")
pickle.dump(reverse_dictionary, reverse_dictionary_file)
reverse_dictionary_file.close()