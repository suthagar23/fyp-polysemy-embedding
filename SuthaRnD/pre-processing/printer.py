import sys


class Printer():
    """
    Print things to stdout on one line dynamically
    """
    processCompletedState = []
    processTotal = []
    totalPercentage = 0
    def initCompletedstate(self, count):
        self.processCompletedState = ["0"] * count
        self.processTotal = [0] * count

    def addValues(self, value, processId):
        self.processTotal[processId] = int(value)
        self.processCompletedState[processId] = "" + str(value) + "%"
        self.totalPercentage = round(sum(self.processTotal)/len(self.processTotal),2)
        if(value!=0):
            sys.stdout.write("\r\x1b[K" + " TOT : "+self.totalPercentage.__str__() + "% | "+ self.processCompletedState.__str__())
            sys.stdout.flush()

    def printNormal(self, completedPercentage):
        sys.stdout.write(
            "\r\x1b[K" + " Completed : " + completedPercentage.__str__() + "%")
        sys.stdout.flush()

    def __init__(self):
        print("Initilized Printer")