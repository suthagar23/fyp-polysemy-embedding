import math
import numpy
import sys
import os

from collections import Counter
from operator import itemgetter
from scipy import linalg


def cosine_sim(vec1, vec2):
    return vec1.dot(vec2) / (linalg.norm(vec1) * linalg.norm(vec2))


def assign_ranks(item_dict):
    ranked_dict = {}
    sorted_list = [(key, val) for (key, val) in sorted(item_dict.items(), key=itemgetter(1), reverse=True)]
    for i, (key, val) in enumerate(sorted_list):
        same_val_indices = []
        for j, (key2, val2) in enumerate(sorted_list):
            if val2 == val:
                same_val_indices.append(j + 1)
        if len(same_val_indices) == 1:
            ranked_dict[key] = i + 1
        else:
            ranked_dict[key] = 1. * sum(same_val_indices) / len(same_val_indices)
    return ranked_dict


def correlation(dict1, dict2):
    avg1 = 1. * sum([val for key, val in dict1.iteritems()]) / len(dict1)
    avg2 = 1. * sum([val for key, val in dict2.iteritems()]) / len(dict2)
    numr, den1, den2 = (0., 0., 0.)
    for val1, val2 in zip(dict1.itervalues(), dict2.itervalues()):
        numr += (val1 - avg1) * (val2 - avg2)
        den1 += (val1 - avg1) ** 2
        den2 += (val2 - avg2) ** 2
    return numr / math.sqrt(den1 * den2)


def spearmans_rho(ranked_dict1, ranked_dict2):
    assert len(ranked_dict1) == len(ranked_dict2)
    if len(ranked_dict1) == 0 or len(ranked_dict2) == 0:
        return 0.
    x_avg = 1. * sum([val for val in ranked_dict1.values()]) / len(ranked_dict1)
    y_avg = 1. * sum([val for val in ranked_dict2.values()]) / len(ranked_dict2)
    num, d_x, d_y = (0., 0., 0.)
    for key in ranked_dict1.keys():
        xi = ranked_dict1[key]
        yi = ranked_dict2[key]
        num += (xi - x_avg) * (yi - y_avg)
        d_x += (xi - x_avg) ** 2
        d_y += (yi - y_avg) ** 2
    return num / (math.sqrt(d_x * d_y))


# Read all the word vectors and normalize them

def read_word_vectors(filename):
    word_vecs = {}
    file_object = open(filename, 'r')

    for line_num, line in enumerate(file_object):
        line = line.strip()
        word = line.split()[0]
        word_vecs[word] = numpy.zeros(len(line.split()) - 1, dtype=float)
        for index, vec_val in enumerate(line.split()[1:]):
            word_vecs[word][index] = float(vec_val)
            # normalize weight vector
        word_vecs[word] /= math.sqrt((word_vecs[word] ** 2).sum() + 1e-6)

    sys.stderr.write("Vectors from: " + filename + " \n")
    return word_vecs


if __name__ == '__main__':
    word_vec_file = sys.argv[1]
    word_sim_file = sys.argv[2]

    word_vecs = read_word_vectors(word_vec_file)
    print('_________________________________________________________________________________')
    print("%15s" % "Num Pairs", "%15s" % "Not found", "%15s" % "Rho")
    print('_________________________________________________________________________________')

    manual_dict, auto_dict = ({}, {})
    not_found, total_size = (0, 0)
    for line in open(word_sim_file, 'r'):
        c = 0
        line = line.strip().lower()
        word1, word2, val = line.split()
        vec_word1 = word1 + "_"
        vec_word2 = word2 + "_"
        word_vec1 = {}
        word_vec2 = {}
        for key in word_vecs:
            if key.startswith(vec_word1) and "GRAM" not in key:
                # print (key)
                c += 1
                word_vec1[key] = word_vecs[key]
                if word1 == word2:
                    word_vec2[key] = word_vecs[key]
            elif key.startswith(vec_word2) and "GRAM" not in key:
                c += 1
                # print(key)
                word_vec2[key] = word_vecs[key]

        if c > 0:
            diff = 1
            for i in word_vec1:
                for j in word_vec2:
                    sim_val = cosine_sim(word_vec1[i], word_vec2[j])
                    if diff == 1:
                        diff = abs(float(sim_val) - float(val))
                        # print(diff)
                        final_sim_val = sim_val
                    elif diff > abs(float(sim_val) - float(val)):
                        diff = abs(float(sim_val) - float(val))
                        # print(diff)
                        final_sim_val = sim_val

                        # print(i)
                    # print(j)
                    # print(sim_val)

            # print (len(sim_list))
            manual_dict[(word1, word2)] = float(val)
            auto_dict[(word1, word2)] = float(final_sim_val)
        elif c == 0:
            not_found += 1
        total_size += 1
    print("%15s" % str(total_size), "%15s" % str(not_found),
          "%15.4f" % spearmans_rho(assign_ranks(manual_dict), assign_ranks(auto_dict)))
