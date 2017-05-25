#!/usr/bin/env python

import numpy as np
import scipy.io as sio
import adaboost


def k_fold(inputs,y,k):
    #k=7
    m=inputs.shape[0]  #total number of data:351
    step=int(m/k)  #50
    dataset=[]
    for i in range(k):
        print(i)
        miniset={}
        if (i+1)!=k:
            #0~50;50~100;
            miniset['testData']=inputs[i*step:(i+1)*step]  
            miniset['testlabel']=y[i*step:(i+1)*step]
            miniset['trainData']=np.concatenate((inputs[0:i*step],inputs[(i+1)*step:]),axis=0)
            miniset['trainlabel']=y[0:i*step]+y[(i+1)*step:]
        else: 
            #300~351
            miniset['testData']=inputs[i*step:]
            miniset['testlabel']=y[i*step:]
            miniset['trainData']=inputs[0:i*step]
            miniset['trainlabel']=y[0:i*step]
        dataset.append(miniset)
    print(len(dataset))
    print(dataset[0]['testData'].shape)
    print(dataset[0]['trainData'].shape)
    print(dataset[6]['testData'].shape)
    print(dataset[6]['trainData'].shape)
    return dataset


data_path = "ionosphereData.mat"
data = sio.loadmat(data_path)
inputs, targets = data['X'], list(map(lambda x: x[0][0], data['Y']))
y=[int(1) if item=='b' else int(-1) for item in targets]

'''
# This is a simple sample.
inputs=np.array([0,1,2,3,4,5,6,7,8,9]).reshape(10,1)
y=[1,1,1,-1,-1,-1,1,1,1,-1]
'''
#classifier,train_errorRate = adaboost.train(inputs,y,150)

dataset=k_fold(inputs,y,k=7)


sum_error_rate=0.0
train_error_rate_list=[]
for miniset in dataset:
    trainData=miniset['trainData']
    trainlabel=miniset['trainlabel']
    testData=miniset['testData']
    testlabel=miniset['testlabel']
    classifiers,train_errorRate = adaboost.train(trainData,trainlabel,150)
    train_error_rate_list.append(train_errorRate)
    predict_result,predict_error_rate=adaboost.predict(testData,testlabel,classifiers)
    print("predict_error_rate is:",predict_error_rate)
    sum_error_rate+=predict_error_rate
average_error_rate=float(sum_error_rate)/len(dataset)
print("average_error_rate is:", average_error_rate)

plot(train_error_rate_list[0],'r*')  
title('TrainError_dataset0')
show()  

plot(train_error_rate_list[1],'b.')  
title('TrainError_dataset1')
show() 

plot(train_error_rate_list[2],'g.')  
title('TrainError_dataset2')
show() 

plot(train_error_rate_list[3],'y*')  
title('TrainError_dataset3')
show() 

plot(train_error_rate_list[4],'r.')  
title('TrainError_dataset4')
show() 

plot(train_error_rate_list[5],'y.')  
title('TrainError_dataset5')
show() 

plot(train_error_rate_list[6],'b*')  
title('TrainError_dataset6')
show() 
