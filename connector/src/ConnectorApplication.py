import json
from pprint import pprint


#load the json file
data = json.load(open('result.txt'))


#initialize needed string
output=""
i=0
for senIndex in range(0,len(data)):
    print i
    i+=1
    wordInfo=data[str(senIndex)]['sentence']['0']['words']['info']
    for wordIndex in range(0,len(wordInfo)):
        if not wordInfo[str(wordIndex)]['isStopWord']:                                          #ignores stop words
            word=wordInfo[str(wordIndex)]['word']+"_"+wordInfo[str(wordIndex)]['posTag']        #integrates word with pos tag
            if output=="":                                                                      #this is to ignore space for the first word
                output+=word
            else:
                output+=(" "+word)
    output+="."
f=open("output.txt","w")
f.write(output.encode('utf8'))
f.close()
