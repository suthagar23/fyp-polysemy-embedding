from nltk import word_tokenize
from nltk.probability import FreqDist


line=open("output.txt").read()
words=word_tokenize(line)
fdist=FreqDist(words)
print(fdist.most_common(10))