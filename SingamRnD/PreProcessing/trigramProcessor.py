


lines=open("trigramOutput.txt").read();
wordset1={}
wordset2={}
i=0
for line in lines.split("\n"):
    if line=="":
        break
    print(i)
    i+=1
    words=line.split("_")
    if words[0] in wordset1.keys():
        if words[1] in wordset1[words[0]]:
            list(wordset2[words[1]]).append(words[2])
        else:
            list(wordset1[words[0]]).append(words[1])
            temp2=[]
            temp2.append(words[2])
            wordset2[words[1]]=temp2
    else:
       temp2=[]
       temp2.append(words[2])
       temp1=[]
       temp1.append(words[1])
       wordset1[words[0]]=temp1
       wordset2[words[1]]=temp2

lines=open("without_pos.txt").read()
strg=""
j=0
stline=True
for line in lines.split(".\n"):
    print(j)
    j+=1
    words=line.split(" ")
    i=0
    while(i<len(words)):
        if(words[i]==""):
            i+=1
            continue
        if(i<len(words)-2):
            if words[i] in wordset1.keys():
                if words[i+1] in wordset1[words[i]]:
                    if words[i+2] in wordset2[words[i+1]]:
                        if(stline):
                            strg += (words[i] + "_" + words[i + 1] + "_" + words[i + 2])
                            stline=False
                        else:
                            strg+=(" "+words[i]+"_"+words[i+1]+"_"+words[i+2])
                        i+=3
                        continue
        if(stline):
            strg+=(words[i])
            stline=False
        else:
            strg += (" "+words[i])
        i+=1
    strg+=".\n"
    stline=True

file=open("text_with_trigram","w")
file.write(strg)
file.close()


