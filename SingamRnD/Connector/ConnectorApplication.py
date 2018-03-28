import json
from pprint import pprint


#load the json file
data = json.load(open('result_trigram.txt'))


#initialize needed string
output=""
i=0
for senIndex in range(0,len(data)):
    print(i)
    i+=1
    wordInfo=data[str(senIndex)]['sentence']['0']['words']['info']
    for wordIndex in range(0,len(wordInfo)):
        if not wordInfo[str(wordIndex)]['isStopWord']:                                          #ignores stop words
            word=wordInfo[str(wordIndex)]['word']+"_"+wordInfo[str(wordIndex)]['posTag']
            if("lemmatize" in wordInfo[str(wordIndex)].keys()):
                if(wordInfo[str(wordIndex)]['posTag'].startswith("N")):
                    word=(wordInfo[str(wordIndex)]['lemmatize']['noun'])+"_N"
                if (wordInfo[str(wordIndex)]['posTag'].startswith("V")):
                    word = (wordInfo[str(wordIndex)]['lemmatize']['verb'])+"_V"
                if (wordInfo[str(wordIndex)]['posTag'].startswith("J")):
                    word = (wordInfo[str(wordIndex)]['lemmatize']['adjective'])+"_J"
                if (wordInfo[str(wordIndex)]['posTag'].startswith("R")):
                    word = (wordInfo[str(wordIndex)]['lemmatize']['adverb'])+"_R"
                if(wordInfo[str(wordIndex)]['posTag'].startswith("T") or wordInfo[str(wordIndex)]['posTag'].startswith("I")):
                    word=""
            if output=="":                                                                      #this is to ignore space for the first word
                output+=word
            else:
                output+=(" "+word)
    output+=".\n"
f=open("result_trigram_preprocessed.txt","w")
f.write(output)
f.close()
