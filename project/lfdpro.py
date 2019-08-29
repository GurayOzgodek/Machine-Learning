import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
import csv

def write_output(predicted_list):
    with open('submission.csv', mode='w') as predicted_file:
        submission = csv.writer(predicted_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        submission.writerow(['ID', 'Predicted'])
        a = 1
        for i in predicted_list:
            submission.writerow([str(a), int(i)])
            a = a + 1

train = pd.read_csv("train.csv")
train = train.drop(['X3','X31','X32','X127','X128','X590' ],axis=1)

test = pd.read_csv("test.csv")
test = test.drop(['X3','X31','X32','X127','X128','X590'],axis=1)

x = train.iloc[:, :train.shape[1]-1]
y = train.iloc[:, train.shape[1]-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.12, random_state=42)
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

lr = LogisticRegression(solver='lbfgs')
lr.fit(x, y)
print("test accuracy {}".format(lr.score(x_test.T, y_test.T)))

x2 = test.iloc[:, :]

predict = lr.predict(x2)

print(predict)

write_output(predict)

pca = PCA(n_components=25)
xtrapca=pca.fit_transform(x)


clf = DecisionTreeClassifier(random_state=5) #her seferinde ver
clf.fit(xtrapca, y)

xtestpca=pca.fit_transform(x2)
ytest=clf.predict(xtestpca).reshape(-1,1)

print(xtestpca.shape)
print(ytest.shape)






'''





x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
#x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, test_size=0.2, random_state=42)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

lr = LogisticRegression(solver="lbfgs")
lr.fit(x_train.T, y_train.T)
predict = lr.predict(x2)

print(predict)



print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))
'''




