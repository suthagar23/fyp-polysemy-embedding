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
from tensorflow.contrib.tensorboard.plugins import projector
import argparse
import sys
from math import sqrt
from six.moves import xrange  # pylint: disable=redefined-builtin
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

parser = argparse.ArgumentParser()
parser.add_argument(
    '--log_dir',
    type=str,
    default=os.path.join(current_path, 'log'),
    help='The log directory for TensorBoard summaries.')
FLAGS, unparsed = parser.parse_known_args()
# Create the directory for TensorBoard variables if there is not.
if not os.path.exists(FLAGS.log_dir):
  os.makedirs(FLAGS.log_dir)

basePath = "/home/singam/Documents/fyp/processing-10/"
histogramInputPath =  basePath + "histogram/"
inputPath = basePath + "pre-processed-final-files/"

embedding_size = 128
vocabulary_size = 80000

dictionary_file = open(histogramInputPath + "top-"+ str(vocabulary_size)+"-words-withIndex-bin", "rb")
dictionary = pickle.load(dictionary_file)

reverse_dictionary_file = open(basePath + "embedding-files/reverse_dictionary-bin", "rb")
reverse_dictionary = pickle.load(reverse_dictionary_file)
batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 2  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
num_sampled = 64  # Number of negative examples to sample.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
valid_size = 100  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
print(valid_examples)

full_examples = [i for i in range(300)]

tf.reset_default_graph()

# Input data.
with tf.name_scope('inputs'):
  valid_dataset = tf.constant(full_examples, dtype=tf.int32)

with tf.device('/cpu:0'):
  # Look up embeddings for inputs.
  with tf.name_scope('embeddings'):
    embeddings = tf.Variable(
      tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

  # Construct the variables for the NCE loss
  with tf.name_scope('weights'):
    nce_weights = tf.Variable(
      tf.truncated_normal(
        [vocabulary_size, embedding_size],
        stddev=1.0 / math.sqrt(embedding_size)))
  with tf.name_scope('biases'):
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, os.path.join(FLAGS.log_dir, 'model.ckpt'))
  print("Model restored.")

  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                            valid_dataset)
  similarity = tf.matmul(
    valid_embeddings, normalized_embeddings, transpose_b=True)

  sim = similarity.eval()




  vec={}
  labels=[]
  final_embeding=normalized_embeddings.eval()
  limit=len(dictionary)
  words=[]
  for i in range(limit):
      words.append(reverse_dictionary[i])
  for word in words:
      index=dictionary[word]
      vector=final_embeding[index]
      labels.append(reverse_dictionary[index])
      vec[word]=vector

  strin=str(limit)+" "+str(embedding_size)+"\n"
  for lable in labels:
      strin+=lable
      for j in vec[lable]:
          strin+=" "+str(j)
      strin+="\n"

  file=open("tensor-vectors.txt","w")
  file.write(strin)
  file.close();
