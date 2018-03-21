


text=open("result_trigram_preprocessed.txt").read();
strg=""
i=1
for sentence in text.split("."):
    for word in sentence.split(" "):
        if("government_NN_0" in word or "government_NN_1" in word):
            print(i)
            i+=1
            word="government_NN_0"
        strg+=" "+word
    strg+=".\n"
f=open("polysemy_after_iterationN+1.txt","w")
f.write(strg)
f.close()