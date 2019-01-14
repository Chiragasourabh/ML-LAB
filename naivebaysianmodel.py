import pandas as pd
msg=pd.read_csv('data6.csv',names=['message','label'])
print('\n Total instances in the dataset: ',msg.shape[0])
msg['labelnum']=msg.label.map({'pos':1,'neg':0})
x=msg.message
y=msg.labelnum

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y)

print('\n Dataset is Split into Training and Testing Samples')
print('\n Training Instances: ',xtrain.shape[0])
print(xtrain)
print('\n Testing Instances :',xtest.shape[0])
print(xtest)

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
xtrain_dtm = count_vect.fit_transform(xtrain)
xtest_dtm = count_vect.transform(xtest)
print('\n Total features extracted using CountVectorizer: ',xtrain_dtm.shape[1])
print('\n Features for first 5 training instances are listed below')
df = pd.DataFrame(xtrain_dtm.toarray(),columns=count_vect.get_feature_names())
print(df[0:5])

from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB().fit(xtrain_dtm,ytrain)
predicted=clf.predict(xtest_dtm)

print('\nClassification Results of Test Dataset are:\n')
for doc, p in zip(xtest,predicted):
    pred = 'pos' if p==1 else 'neg'
    print('%s -->  %s '%(doc,pred))

from sklearn import metrics
print('\nAccuracy of the classifier is',metrics.accuracy_score(ytest,predicted))
print('\nConfusion Matrix')
print(metrics.confusion_matrix(ytest,predicted))
print('\nRecall and Precision')
print(metrics.recall_score(ytest,predicted))
print(metrics.precision_score(ytest,predicted))
