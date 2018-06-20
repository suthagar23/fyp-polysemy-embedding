import json
from datetime import datetime
import re
import os


basePath = "/home/suthagar/Desktop/PiraveenaFiles/"
firstPath =  basePath + "adjacency_clusters_0/"
firstOut = basePath + "adjacency_clusters_0/output/"

startTime = datetime.now()
print("Process started : ", startTime)
all_files = os.listdir(firstPath)
if len(all_files ) >0 :
    if not os.path.exists(firstOut):
        os.makedirs(firstOut)
for inputFileName in all_files:
    if os.path.isfile(firstPath + inputFileName):
        print(" Started File", " - ", inputFileName)
        inputData = json.load(open(firstPath + inputFileName))
        outputData = {}
        for key, value in inputData.items():
            if value not in outputData:
                outputData[value] = 1
            else:
                outputData[value] += 1
        with open(firstOut + inputFileName, 'w+') as f:
            json_data = json.dumps(outputData)
            f.write(json_data)

inputPaths = []
outputPaths = []
TOTAL_INTERATIONS = 9

for itr in range(0, TOTAL_INTERATIONS):
    inputPaths.append(basePath + "adjacency_clusters_" + str(itr) + "/")
    outputPaths.append(basePath + "adjacency_clusters_" + str(itr) + "/output/")
print(inputPaths)
print(outputPaths)

for inputPathId in range(1, len(inputPaths)):
    all_files = os.listdir(inputPaths[inputPathId])
    if len(all_files) > 0:
        if not os.path.exists(outputPaths[inputPathId]):
            os.makedirs(outputPaths[inputPathId])

    for inputFileName in all_files:
        if os.path.isfile(inputPaths[inputPathId] + inputFileName):
            print(" Started File", " - ", inputPaths[inputPathId] + inputFileName)
            inputData = json.load(open(inputPaths[inputPathId] + inputFileName))

            previousCountData = json.load(open(inputPaths[inputPathId-1] + "output/"+ inputFileName))
            print(" Previous File - ", inputPaths[inputPathId-1] + inputFileName)

            outputData = {}
            for key, value in inputData.items():
                if value not in outputData:
                    outputData[value] = previousCountData[value]

                else:
                    outputData[value] += 1
            with open(outputPaths[inputPathId] + inputFileName, 'w+') as f:
                json_data = json.dumps(outputData)
                f.write(json_data)

endTime = datetime.now()
print("\nProcess Stopped : ", endTime)
print("\nDuration : ", endTime - startTime)