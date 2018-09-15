import json;
basePath="data/"
inputFile="vectors.txt"
targetWordsFile="target-words.txt"
#Create a text file names as target-words.txt which has the list of words(with pos tag) that need to be clustered.
outputfile="targetvectors"

targetwords=set(open(basePath+targetWordsFile,"r").read().split("\n"));

words={}
with open(basePath+inputFile) as vectors:
    for vector in vectors:
        vec=vector.split(" ")
        parts=vec[0].split("_")
        if (len(parts)==3):
            word="_".join([parts[0],parts[1]])
            if(word in targetwords):
                w=vec.pop(0)
                if(word in words):
                    words[word][w]=" ".join(vec)
                else:
                    dict={}
                    dict[w]=" ".join(vec)
                    words[word]=dict

results = json.dumps(words)

file=open(basePath+outputfile,"w")
file.write(results)
file.close()

print("finished")

