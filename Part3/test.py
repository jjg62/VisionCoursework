import numpy as np
import main
from os import listdir
import time

#Directory names for test images an annotations
testDirectory = "test2"
annotationsDirectory = "annotations2"
#Annotation folder must contain .txt files with SAME NAME as test files

testImageNames = listdir(testDirectory)

#Get features from training images
print("Training...")
main.train()

#Store FPR, TPR, ACC and time taken
fprs = np.zeros(len(testImageNames))
tprs = np.zeros(len(testImageNames))
accs = np.zeros(len(testImageNames))
times = np.zeros(len(testImageNames))

i = 0
for name in testImageNames:
    #Get expected results from annotations folder
    actualLabels = []
    f = open(annotationsDirectory + "/" + name[:-4] + ".txt")  # Open matching annotations file
    lines = f.readlines()
    for line in lines:
        actualLabels.append(line.partition(",")[0])

    print("\nTesting " + name)

    startTime = time.time()

    #Run the program on the test image
    foundLabels = main.test(testDirectory + "/" + name, True)

    #Time taken will only be accurated when show = False
    times[i] = time.time() - startTime


    #False positives
    print("False Positives:")
    falsePositives = 0
    for found in foundLabels:
        if found not in actualLabels:
            print(found)
            falsePositives += 1

    print("( FPR:", falsePositives / (50-len(actualLabels)), ")")
    fprs[i] = (falsePositives / (50-len(actualLabels)))

    #False Negatives
    print("False Negatives: ")
    falseNegatives = 0
    for actual in actualLabels:
        if actual not in foundLabels:
            print(actual)
            falseNegatives += 1

    print("( TPR:", (len(actualLabels) - falseNegatives) / len(actualLabels), ")")
    tprs[i] = (len(actualLabels) - falseNegatives) / len(actualLabels)
    accs[i] = (len(actualLabels) - falseNegatives + (50-len(actualLabels)) - falsePositives) / 50

    i += 1

print("\n\nOVERALL AVERAGES:")
print("TPR:", tprs.mean())
print("FPR:", fprs.mean())
print("ACC:", accs.mean())
print("Runtime:" , times.mean())

