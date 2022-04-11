import main
from os import listdir
import time

testImageNames = listdir('scale_test')


totalFalseNegatives = 0
totalFalsePositives = 0

for testImageName in testImageNames:

    #Read the text file containing actual labels
    actualLabels = []
    f = open("annotations/" + testImageName[:-4] + ".txt") #Open annotations file
    lines = f.readlines()
    for line in lines:
        actualLabels.append(line.partition(",")[0])

    startTime = time.time()

    print("\nTesting file: ", testImageName)

    #Run the program on this test image
    matches = main.recognize(testImageName, True)


    foundLabels = []

    #Find false positives - found labels that weren't in the actual list
    print("False Positives: ")
    for match in matches:
        matchName = match.instanceName[4:-4]
        foundLabels.append(matchName)
        if(matchName not in actualLabels):
            totalFalsePositives += 1
            print(matchName)

    #Find false negatives: actual labels that weren't found
    print("False Negatives: ")
    for actual in actualLabels:
        if(actual not in foundLabels):
            totalFalseNegatives += 1
            print(actual)


    accuracy = 1 - (totalFalseNegatives + totalFalsePositives)/50
    print("Accuracy: ", accuracy)
    print("TIME TAKEN: ", time.time() - startTime, "\n")

    totalFalseNegatives = 0
    totalFalsePositives = 0
