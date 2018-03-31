
data=open("result_trigram_preprocessed.txt").read()

sentences=data.split(".\n ")

result=""
i=0;
for text in sentences:
    i+=1
    print(i)
    result+=(text+".\n")

file=open("result_trigram_preprocessed.txt","w")
file.write(result)
file.close()

