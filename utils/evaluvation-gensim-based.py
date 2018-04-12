from gensim.models import Word2Vec,KeyedVectors
# load the dump as gensim model
model=KeyedVectors.load_word2vec_format('model-dumps/tensor-vectors.txt', binary=False);
# evaluvate with the given data set
print(model.wv.evaluate_word_pairs('datasets/wordsim-353-with-pos.txt'));