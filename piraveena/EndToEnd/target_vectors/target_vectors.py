import json;
basePath=""
inputFile=""
targetWordsFile=""
#Create a text file names as target-words.txt which has the list of words(with pos tag) that need to be clustered.
outputfile=""

def config_target_vectors(base,input,targetwords,output):
    global basePath
    global inputFile
    global targetWordsFile
    # Create a text file names as target-words.txt which has the list of words(with pos tag) that need to be clustered.
    global outputfile

    basePath = base
    inputFile = input
    targetWordsFile = targetwords
    # Create a text file names as target-words.txt which has the list of words(with pos tag) that need to be clustered.
    outputfile = output



def get_needed_vectors():
    targetwords=set(open(basePath+targetWordsFile,"r").read().split("\n"));
    print (len(targetwords))

    words={}
    with open(basePath+inputFile) as vectors:
        for vector in vectors:
            vec=vector.split(" ")
            parts=vec[0].split("_")
            # print (len(parts))
            word = ""
            for x in range(len(parts)):
                # print(x)
                if x == 0:
                    word = parts[x]
                else:
                    word= word + "_" + parts[x]
            # print ("In vectors: "+ word)
            if(word in targetwords):
                # print ("In target vectors: " + word)
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

