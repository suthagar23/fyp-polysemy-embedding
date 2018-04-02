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

basePath = "/media/suthagar/Data/Corpus/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/tmp/"
histogramInputPath =  basePath + "histogram/"
inputPath = basePath + "pre-processed-final-files/"

embedding_size = 128
vocabulary_size = 80000

dictionary_file = open(histogramInputPath + "top-"+ str(vocabulary_size)+"-words-withIndex-bin", "rb")
dictionary = pickle.load(dictionary_file)

reverse_dictionary_file = open(basePath + "embedding-files/reverse_dictionary-bin", "rb")
reverse_dictionary = pickle.load(reverse_dictionary_file)


# data_file = open(basePath + "embedding-files/data-bin", "rb")
# data = pickle.load(data_file)


# Step 4: Build and train a skip-gram model.
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
print(full_examples[:10], len(full_examples))

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



  # final_embeddings = normalized_embeddings.eval()
  # tsne = TSNE(
  #     perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
  # plot_only = 500
  # low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  # print(low_dim_embs[0])

  print(len(sess.run(embeddings)), len(sess.run(embeddings)[0]), sess.run(embeddings)[0][:10])
  # print(len(sess.run(norm)),sess.run(norm)[0])
  # print(len(sess.run(normalized_embeddings)),sess.run(normalized_embeddings)[0])
  # print(len(sess.run(valid_embeddings)),sess.run(valid_embeddings)[0])
  # print(len(sess.run(similarity)),sess.run(similarity)[0])


  sim = similarity.eval()
  print("SIM******")
  # print(-sim[95])
  # print(len(-sim[95]))
  # # print(((-sim[95, :])[1:9]))
  # print(((-sim[95, :])[1:9])[:8])
  # print(((-sim[95, :]).argsort()[1:9])[:8])

  for i in xrange(95, 100):
    # print(i, valid_examples[i], reverse_dictionary[i], reverse_dictionary[valid_examples[i]])
    if full_examples[i] < len(reverse_dictionary):
      valid_word = reverse_dictionary[full_examples[i]]
      top_k = 8  # number of nearest neighbors
      nearest = (-sim[i, :]).argsort()[1:top_k + 1]
      print(i, nearest[:8])
      log_str = 'Nearest to %s:' % valid_word
      for k in xrange(top_k):
        if nearest[k] < len(reverse_dictionary):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
      print(log_str)


  # word_list = ["address_N"]
  # print("\n\n", word_list)
  # for valid_word in word_list:
  #   print(valid_word)
  #   i = dictionary.get(valid_word)
  #   print(i)
  #
  #   top_k = 8  # number of nearest neighbors
  #   nearest = (-sim[i, :]).argsort()[1:top_k + 1]
  #   log_str = 'Nearest to %s:' % valid_word
  #   for k in xrange(top_k):
  #     if nearest[k] < len(reverse_dictionary):
  #       close_word = reverse_dictionary[nearest[k]]
  #       log_str = '%s %s,' % (log_str, close_word)
  #   print(log_str)




print("loaded")




#134321985 - len data
#
N=100
K=8
MAX_ITERS = 1000
start = time.time()
ShapeY = 1

points = tf.Variable(tf.random_uniform([N, ShapeY]))
cluster_assignments = tf.Variable(tf.zeros([N], dtype=tf.int64))
# print(sess.run(points))
# print(sess.run(cluster_assignments))

# Silly initialization:  Use the first K points as the starting
# centroids.  In the real world, do this better.
centroids = tf.Variable(tf.slice(points.initialized_value(), [0,0], [K,ShapeY]))

# Replicate to N copies of each centroid and K copies of each
# point, then subtract and compute the sum of squared distances.
rep_centroids = tf.reshape(tf.tile(centroids, [N, 1]), [N, K, ShapeY])
rep_points = tf.reshape(tf.tile(points, [1, K]), [N, K, ShapeY])
sum_squares = tf.reduce_sum(tf.square(rep_points - rep_centroids),
                            reduction_indices=2)

# Use argmin to select the lowest-distance point
best_centroids = tf.argmin(sum_squares, 1)
did_assignments_change = tf.reduce_any(tf.not_equal(best_centroids,
                                                    cluster_assignments))

def bucket_mean(data, bucket_ids, num_buckets):
    total = tf.unsorted_segment_sum(data, bucket_ids, num_buckets)
    count = tf.unsorted_segment_sum(tf.ones_like(data), bucket_ids, num_buckets)
    return total / count

means = bucket_mean(points, best_centroids, K)


# Do not write to the assigned clusters variable until after
# computing whether the assignments have changed - hence with_dependencies
with tf.control_dependencies([did_assignments_change]):
    do_updates = tf.group(
        centroids.assign(means),
        cluster_assignments.assign(best_centroids))

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print("********* 111111111 ************")
print(len(sess.run(points)), len(sess.run(points)[1]), sess.run(points)[1][:10])
print(len(sess.run(cluster_assignments)),sess.run(cluster_assignments)[:20])

changed = True
iters = 0

while changed and iters < MAX_ITERS:
    iters += 1
    [changed, _] = sess.run([did_assignments_change, do_updates])


[centers, assignments] = sess.run([centroids, cluster_assignments])
end = time.time()
print(("Found in %.2f seconds" % (end-start)), iters, "iterations")
print( "Centroids:")
print( centers)
print("Cluster assignments:", assignments)

# print(len(assignments), assignments[:20])





# def create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed):
#     np.random.seed(seed)
#     slices = []
#     centroids = []
#     # Create samples for each cluster
#     for i in range(n_clusters):
#         samples = tf.random_normal((n_samples_per_cluster, n_features),
#                                    mean=0.0, stddev=5.0, dtype=tf.float32, seed=seed, name="cluster_{}".format(i))
#         current_centroid = (np.random.random((1, n_features)) * embiggen_factor) - (embiggen_factor/2)
#         centroids.append(current_centroid)
#         samples += current_centroid
#         slices.append(samples)
#     # Create a big "samples" dataset
#     samples = tf.concat(slices, 0, name='samples')
#     centroids = tf.concat(centroids, 0, name='centroids')
#     return centroids, samples
#
# n_features = 2
# n_clusters = 30
# n_samples_per_cluster = 500
# seed = 700
# embiggen_factor = 70
#
# np.random.seed(seed)
#
# centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)
#
#
# model = tf.global_variables_initializer()
# with tf.Session() as session:
#     sample_values = session.run(samples)
#     centroid_values = session.run(centroids)
#     print(centroid_values[0])
#     print(len(sample_values))
#     print((sample_values[0]))
#     print((sample_values[1]))
#     print((sample_values[1]))
#     print((sample_values[1499]))
#
# def plot_clusters(all_samples, centroids, n_samples_per_cluster):
#   import matplotlib.pyplot as plt
#   # Plot out the different clusters
#   # Choose a different colour for each cluster
#   colour = plt.cm.rainbow(np.linspace(0, 1, len(centroids)))
#   for i, centroid in enumerate(centroids):
#     # Grab just the samples fpr the given cluster and plot them out with a new colour
#     samples = all_samples[i * n_samples_per_cluster:(i + 1) * n_samples_per_cluster]
#     # print(i, len(samples[:, 0]), len(samples[:, 1]))
#
#
#     plt.scatter(samples[:, 0], samples[:, 1], c=colour[i])
#     # Also plot centroid
#     plt.plot(centroid[0], centroid[1], markersize=35, marker="x", color='k', mew=10)
#     plt.plot(centroid[0], centroid[1], markersize=30, marker="x", color='m', mew=5)
#   plt.show()

# plot_clusters(sample_values, centroid_values, n_samples_per_cluster)


import tensorflow as tf
from random import choice, shuffle
from numpy import array


def TFKMeansCluster(vectors, noofclusters):
    """
    K-Means Clustering using TensorFlow.
    'vectors' should be a n*k 2-D NumPy array, where n is the number
    of vectors of dimensionality k.
    'noofclusters' should be an integer.
    """

    noofclusters = int(noofclusters)
    assert noofclusters < len(vectors)

    # Find out the dimensionality
    dim = len(vectors[0])

    # Will help select random centroids from among the available vectors
    vector_indices = list(range(len(vectors)))
    shuffle(vector_indices)

    # GRAPH OF COMPUTATION
    # We initialize a new graph and set it as the default during each run
    # of this algorithm. This ensures that as this function is called
    # multiple times, the default graph doesn't keep getting crowded with
    # unused ops and Variables from previous function calls.

    graph = tf.Graph()

    with graph.as_default():

        # SESSION OF COMPUTATION

        sess = tf.Session()

        ##CONSTRUCTING THE ELEMENTS OF COMPUTATION

        ##First lets ensure we have a Variable vector for each centroid,
        ##initialized to one of the vectors from the available data points
        centroids = [tf.Variable((vectors[vector_indices[i]]))
                     for i in range(noofclusters)]
        ##These nodes will assign the centroid Variables the appropriate
        ##values
        centroid_value = tf.placeholder("float64", [dim])
        cent_assigns = []
        for centroid in centroids:
            cent_assigns.append(tf.assign(centroid, centroid_value))

        ##Variables for cluster assignments of individual vectors(initialized
        ##to 0 at first)
        assignments = [tf.Variable(0) for i in range(len(vectors))]
        ##These nodes will assign an assignment Variable the appropriate
        ##value
        assignment_value = tf.placeholder("int32")
        cluster_assigns = []
        for assignment in assignments:
            cluster_assigns.append(tf.assign(assignment,
                                             assignment_value))

        ##Now lets construct the node that will compute the mean
        # The placeholder for the input
        mean_input = tf.placeholder("float", [None, dim])
        # The Node/op takes the input and computes a mean along the 0th
        # dimension, i.e. the list of input vectors
        mean_op = tf.reduce_mean(mean_input, 0)

        ##Node for computing Euclidean distances
        # Placeholders for input
        v1 = tf.placeholder("float", [dim])
        v2 = tf.placeholder("float", [dim])
        euclid_dist = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(
            v1, v2), 2)))

        ##This node will figure out which cluster to assign a vector to,
        ##based on Euclidean distances of the vector from the centroids.
        # Placeholder for input
        centroid_distances = tf.placeholder("float", [noofclusters])
        cluster_assignment = tf.argmin(centroid_distances, 0)

        ##INITIALIZING STATE VARIABLES

        ##This will help initialization of all Variables defined with respect
        ##to the graph. The Variable-initializer should be defined after
        ##all the Variables have been constructed, so that each of them
        ##will be included in the initialization.
        init_op = tf.initialize_all_variables()

        # Initialize all variables
        sess.run(init_op)

        ##CLUSTERING ITERATIONS

        # Now perform the Expectation-Maximization steps of K-Means clustering
        # iterations. To keep things simple, we will only do a set number of
        # iterations, instead of using a Stopping Criterion.
        noofiterations = 100
        for iteration_n in range(noofiterations):

            ##EXPECTATION STEP
            ##Based on the centroid locations till last iteration, compute
            ##the _expected_ centroid assignments.
            # Iterate over each vector
            for vector_n in range(len(vectors)):
                vect = vectors[vector_n]
                # Compute Euclidean distance between this vector and each
                # centroid. Remember that this list cannot be named
                # 'centroid_distances', since that is the input to the
                # cluster assignment node.
                distances = [sess.run(euclid_dist, feed_dict={
                    v1: vect, v2: sess.run(centroid)})
                             for centroid in centroids]
                # Now use the cluster assignment node, with the distances
                # as the input
                assignment = sess.run(cluster_assignment, feed_dict={
                    centroid_distances: distances})
                # Now assign the value to the appropriate state variable
                sess.run(cluster_assigns[vector_n], feed_dict={
                    assignment_value: assignment})

            ##MAXIMIZATION STEP
            # Based on the expected state computed from the Expectation Step,
            # compute the locations of the centroids so as to maximize the
            # overall objective of minimizing within-cluster Sum-of-Squares
            for cluster_n in range(noofclusters):
                # Collect all the vectors assigned to this cluster
                assigned_vects = [vectors[i] for i in range(len(vectors))
                                  if sess.run(assignments[i]) == cluster_n]
                # Compute new centroid location
                new_location = sess.run(mean_op, feed_dict={
                    mean_input: array(assigned_vects)})
                # Assign value to appropriate variable
                sess.run(cent_assigns[cluster_n], feed_dict={
                    centroid_value: new_location})

        # Return centroids and assignments
        centroids = sess.run(centroids)
        assignments = sess.run(assignments)
        return centroids, assignments

TFKMeansCluster(sess.run(points),10)