import os


basepath="/home/singam/Documents/fyp/processing-10/"
inputfolder="pre-processed-final-files/"
ouputfile="singlefile/data.txt"
directories = [x[0] for x in os.walk(basepath+"/"+inputfolder)] [1:];
i=1
with open(basepath+ouputfile,"a") as out:
    for directory in directories:
        print(i);
        i+=1;
        for file in os.listdir(directory):
            if file.endswith(".txt"):
                data=open(directory+"/"+file,"r").read()+"\n";
                out.write(data);
out.close();



