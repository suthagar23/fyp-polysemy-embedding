


text=open("polysemy_after_iteration5.txt","r").read()
map={}
for line in text. split(".\n"):
    for word in line.split(" "):
        if "government_NN" in word:
            if word in map.keys():
                map[word]+=1
            else:
                map[word]=1

strg=""
for key in map.keys():
    print(key,"    :   ",map[key])
    strg+=(key+"  :  "+str(map[key])+"\n")
f=open("distribution_iteration5.txt","w")
f.write(strg)
f.close()
