from datetime import datetime

def replace(base,inputFile,target,output):
    basepath = base
    inputfile = inputFile
    targetwordfile = target
    outputfile = output+"/data-no-dub.txt"
    outputdubfile = output+"/data-dub.txt"

    output = open(basepath + outputfile, "w");
    outputdub = open(basepath + outputdubfile, "w");
    targetwords = open(basepath + targetwordfile, "r").read().split("\n");
    count = {};

    for word in targetwords:
        count[word] = 1;
    startTime = datetime.now();
    i = 1;
    with open(basepath+inputfile,"r") as file:
        for line in file:
            print(i)
            i+=1;
            line=line.split(".")[0]
            containtargetword=False
            firstword=True
            st="";
            for word in line.split(" "):
                if(word==""):
                    continue;
                if word in targetwords:
                    containtargetword=True;
                    if(firstword):
                        st+=(word+"_"+str(count[word]))
                        firstword=False;
                    else:
                        st += (" "+word + "_" + str(count[word]))
                    count[word]+=1;
                else:
                    if(firstword):
                        st+=word
                        firstword=False;
                    else:
                        st+=(" "+word)
            st+="\n"
            if(containtargetword):
                for j in range(11):
                    outputdub.write(st)
            else:
                outputdub.write(st)
            output.write(st)
            firstword=True
            containtargetword=False
    output.close();
    outputdub.close();
    endtime=datetime.now();
    print(endtime-startTime)

replace("/home/singam/Documents/fyp/sagemakerfiles/","singlefile/data.txt","target-words.txt","singlefile")