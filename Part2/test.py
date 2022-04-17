import main
from os import listdir
import time



#Directory with all test files
testDir = 'test'
#Annotation / test result directory
annotationDir = 'annotations'
#Annotation folder must contain .txt files with SAME NAME as test files

testImageNames = listdir(testDir)

#For each test image
for testImageName in testImageNames:

    falseNegatives = 0
    falsePositives = 0

    #Read the text file containing actual labels
    actualLabels = []
    f = open("annotations/" + testImageName[:-4] + ".txt") #Open annotations file
    lines = f.readlines()
    for line in lines:
        #Get names of instances
        actualLabels.append(line.partition(",")[0])

    startTime = time.time()

    print("\nTesting file: ", testImageName)

    #Run the program on this test image
    matches = main.recognize(testDir + "/" + testImageName, True)
    #(Time will only be accurate when show=False)

    foundLabels = []

    #Find false positives - found labels that weren't in the actual list
    print("False Positives: ")
    for match in matches:
        matchName = match.instanceName[4:-4]
        foundLabels.append(matchName)
        if(matchName not in actualLabels):
            falsePositives += 1
            print(matchName)

    #Find false negatives: actual labels that weren't found
    print("False Negatives: ")
    for actual in actualLabels:
        if(actual not in foundLabels):
            falseNegatives += 1
            print(actual)


    accuracy = (len(actualLabels) - falseNegatives + (50-len(actualLabels)) - falsePositives) / 50
    print("Accuracy: ", accuracy)
    print("TIME TAKEN: ", time.time() - startTime, "\n")

