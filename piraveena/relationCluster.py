import collections
import matplotlib
import json
import os

from nltk.corpus import wordnet as wn
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
matplotlib.use('Agg')

path = "/home/piraveena/fyp_data/"
modelPath = path + "/model2/"
targetvector = path + "/targetvectors-2/"
output_image_path = "relations/images/4/"
output_cluster_path ="relationCluster/4/"
similar_senses ="similarsenses/"

if not os.path.exists(output_cluster_path):
    os.makedirs(output_cluster_path)

if not os.path.exists(output_image_path):
    os.makedirs(output_image_path)
if not os.path.exists(similar_senses):
    os.makedirs(similar_senses)
def no_of_senses(lemma):
    if lemma.isalpha():
        senses = wn.synsets(lemma)
    return len(senses)

# def get_similar_words():
#     file = modelPath + "vectors.txt"
#     print("start building the model")
#     model = KeyedVectors.load_word2vec_format(file, binary=False);

def get_relation(lemma, numbOfSenses):
    file = modelPath + "vectors.txt"
    print ("start building the model")
    model = KeyedVectors.load_word2vec_format(file, binary=False);
    cluster_file_path = output_cluster_path +  lemma + "cluster.txt"
    print("opening o/p file")

    cluster_file = open(cluster_file_path, "w+")

    firstoutput = {}
    # numbOfSenses = no_of_senses(lemma)
    list_similar =[]
    # similar_senses_file = similar_senses + lemma + "similar.txt"
    # similar_file = open(similar_senses_file, "w+")
    for j in range (0, numbOfSenses):

        word1 = lemma + "_" + str(j)

        listOfSimilar = []
        for i in range (j+1,numbOfSenses):
            word2 = lemma+"_"+str(i)
            similarity = model.similarity(word1, word2)
            # print(word1, word2 + ":" + str(similarity))
            if (similarity>0.55):
                # print (word1, word2)
                listOfSimilar.append(word2)
        listOfSimilar.append(word1)
        firstoutput[word1] = listOfSimilar
        listx = model.most_similar(positive=word1, topn=20)
        print (word1)
        print (listx)
        list_similar.append(listx)

    # similar_file.write (list_similar)
    # similar_file.close()
    # print (firstoutput)
    # print("rock_N_1", model.most_similar(positive=["rock_N_1"]))
    # print("rock_N_4", model.most_similar(positive=["rock_N_4"]))
    # print (model.similarity("rock_N_1", "rock_N_4"))
    # print(model.similarity("rock_N_3", "rock_N_5"))
    output = {}
    i=0
    keys = list(firstoutput.keys())
    for key in keys:
        new_sense = lemma + "_" + str(i)
        dict=[]
        status=False
        for j in firstoutput[key]:
            # if(j not in output.keys()):

                # print (str(keys.index(key)))

            if(j in output):
                new_sense=output[j]
                status=True
                break;
        for j in firstoutput[key]:
            output[j] = new_sense
        if(not status):
            i+=1
        # print(output)
    # print (output)
    cluster_file.write(json.dumps(output))
    cluster_file.close()

def plot(lemma):
    # list of vectors for lemmas
    vecs = []
    # list of lemma
    words = []

    file = targetvector
    data = json.load(open(file)).get(lemma)
    for key in data:
        words.append(key)
        word_vec = []
        vector_list = (data[key].split("\n"))[0].split(" ")[:-1]
        for i in (vector_list):
            word_vec.append(float(i))
        vecs.append(word_vec)

    no_of_senses = len(words)
    return vecs, words,no_of_senses


x= no_of_senses("address")
lemma = "address_V"
senses = plot(lemma)
get_relation(lemma ,senses)
