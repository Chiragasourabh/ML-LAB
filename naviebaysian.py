import pandas as pd
import numpy as np
data = pd.DataFrame(pd.read_csv('data5.csv'))
training=np.array(data.iloc[:,:])
testdata=np.array(data.iloc[10:,:])
count=0
def counting(inputlist,j,st,count):
    if(inputlist[j][4]==st):
        count=count+1
    return count
def find(inputlist,col,match):
    count = 0
    for row in inputlist:
        if row[col]==match:
            count=count+1
    return count
dataYes=[row for row in training if row[4]=="Yes" ]
dataNo=[row for row in training if row[4]=="No" ]
tpYes=len(dataYes)/(len(training))
tpNo=len(dataNo)/(len(training))
for j in range(len(testdata)):
    print(testdata[j][:-1])
    yes=1
    no=1
    for d in range(4):
        yes=yes*(find(dataYes,d,testdata[j][d])/len(dataYes))
        no=no*(find(dataNo,d,testdata[j][d])/len(dataNo))
    resultyes = yes*tpYes
    resultno = no*tpNo
    print("Probability of Yes: ",resultyes)
    print("Probability of No: ",resultno)
    if resultyes>resultno:
        print("Classified as YES\n")
        count=counting(testdata,j,'Yes',count)
    else:
        print("Classified as NO\n")
        count=counting(testdata,j,'No',count)
print("\nAccuracy of the Classifier is: ",count/len(testdata))
