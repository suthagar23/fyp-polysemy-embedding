
data=open("output.txt").read().decode('utf8')

sentences=data.split(". ")

result=""
i=0;
for text in sentences:
    i+=1
    print(i)
    result+=(text+".\n")

file=open("sentence.txt","w")
file.write(result.encode('utf8'))
file.close()

