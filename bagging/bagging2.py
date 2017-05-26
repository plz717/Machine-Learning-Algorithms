import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from pylab import *


def dataset_gen(n=20):
    df=pd.read_csv('student-por1.csv',nrows=600)  #read csv data into dataframe
    print("df.shape:",df.shape)
    number_samples=df.shape[0]  #number of the total data for training
    train_subsets=[]  #to store n training sets
    #test_subsets=[]
    for i in range(n):
        rows=np.random.choice(df.index.values,number_samples)  #randomly select number_samples samples from the whole 600 samples
        sampled_df=df.ix[rows] 
        #test_df=df.drop(rows)
        train_subsets.append(sampled_df)
        #test_subsets.append(test_df)
    print("{} subsets in total.".format(len(train_subsets)))
    #print("{} subsets in total.".format(len(test_subsets)))
    return train_subsets


def normalize(X):
    #normalize and standaize the input data
    normalized_X=preprocessing.normalize(X)
    standardized_X=preprocessing.scale(X)
    return standardized_X

def train(train_dataset):
    train_X=train_dataset.iloc[:,0:32]  #0~31:features of the samples
    train_y=train_dataset.iloc[:,32]  #32:labels of the samples
    train_X=normalize(train_X)

    model = DecisionTreeRegressor()  #build a CART tree
    model.fit(train_X,train_y)  #using the data to fit the model
    
    return model

trainset=dataset_gen(20)
test_df=pd.read_csv('student-por1.csv')
test_X=test_df.iloc[601:,0:32]
test_X=normalize(test_X)  #test input
test_y=test_df.iloc[601:,32]  #label for test input
predicted_final=np.zeros(len(test_y))   #initialize final predicted result
for i in range(20):   #for every weak classifier
    model=train(trainset[i])  #train the weak classifier
    predicted=model.predict(test_X)  #using the classifier to predict 
    predicted_final+=np.array(predicted)  #sum of the result from all classifiers
predicted_final=predicted_final/20  #final predict result 
mse_error=((predicted_final-test_y)*(predicted_final-test_y)).sum()/len(test_y)  #MSE error between  predicted result and label
print("predicted is::",predicted_final)
print("label is:",test_y.values)
print("predicted-label:",predicted_final-test_y.values)  #difference between predicted result and label
print("final mse_error is:{}".format(mse_error))
plot(predicted_final,'b-')
plot(test_y.values,'y-')
title("predict and label")
show()

