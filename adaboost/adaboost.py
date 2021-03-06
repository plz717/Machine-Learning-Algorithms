import numpy as np



def weak_classify(data_array,dim,threshold,sign):
    weak_result=np.ones((data_array.shape[0],1))  
    if sign=='<':
        weak_result[data_array[:,dim]<=threshold]=-1.0
    else:
        weak_result[data_array[:,dim]>threshold]=-1.0
    return weak_result   #weak_result:列向量
    
    
def best_weak_classify(data_array,labels,D):
    '''data_array:np.array   labels:列向量   D：列向量'''
    m=data_array.shape[0]
    n=data_array.shape[1]
    
    labels_array=np.array(labels).reshape(len(labels),1)
    numstep=10.0
    best_weak_classifier={} 
    best_weak_result=np.zeros((m,1))
    minerror=np.inf
    for i in range(n):
        feature_min=data_array[:,i].min()
        feature_max=data_array[:,i].max()
        step=(feature_max-feature_min)/numstep
        for j in range(-1,int(numstep)+1):
            for sign in ['<','>']:
                threshold=feature_min+step*float(j)
                weak_result=weak_classify(data_array,i,threshold,sign)  #weak_result:列向量
                error=np.zeros((m,1))
                error[weak_result!=labels_array]=1
                #print("error is :",error)
                # print("weak_result is:",weak_result)
                # print("labels is:",labels)
                # print("D.T shape:{},error shape:{}".format(D.T.shape,error.shape))

                weighted_error_array=np.dot(D.T,error)  #行向量×列向量=数
                weighted_error=weighted_error_array[0][0]
                #print("weigthed error:{}".format(weighted_error))
                
                if weighted_error<minerror:
                    minerror=weighted_error
                    best_weak_result=weak_result.copy()
                    best_weak_classifier['dim']=i
                    best_weak_classifier['threshold']=threshold
                    best_weak_classifier['sign']=sign
    return minerror,best_weak_result,best_weak_classifier


def train(data_array,labels,num_iter):
    labels_array=np.array(labels).reshape(len(labels),1)
    m=data_array.shape[0]  #number of samples
    n=data_array.shape[1]  #number of features
    D=np.ones((m,1))/m  #D:列向量
    weakClassifiers=[]
    AggreClassifier=np.zeros((m,1))
    errorRate_list=[]
    for i in range(num_iter):
        weak_error,weak_result,best_weak_classifier=best_weak_classify(data_array,labels,D)  #weak_result:列向量
        # alpha=float(1/2*np.log((1-weak_error)/weak_error))
        alpha = float(0.5*np.log((1.0-weak_error) / (weak_error+1e-15)))
        best_weak_classifier['alpha']=alpha
        weakClassifiers.append(best_weak_classifier)
        weightD=(-1*alpha*labels_array)*weak_result #weightD should be a 列向量!!
        D=D*np.exp(weightD)   #列向量*列向量=列向量
        Zm=D.sum()  
        #print("Zm is:",Zm)
        D=D/Zm
        #print("D is:{}".format(D))
        AggreClassifier+=weak_result*alpha
        AggreError=np.zeros((m,1))
        AggreError[np.sign(AggreClassifier)!=labels_array]=1
        #print("np.sign(AggreClassifier) is:",np.sign(AggreClassifier))
        #print("labels array is:",labels_array)
        #print("aggre error is:",AggreError)
        error_rate=AggreError.sum()/m
        #print("total error: ",error_rate)
        errorRate_list.append(error_rate)
        if error_rate == 0.0:
            break
    return weakClassifiers, errorRate_list


def adaboostclassify(data_array,classifiers):
    aggre_classify_result=np.zeros((data_array.shape[0],1))  #初始化集成分类器分类结果
    for i in range(len(classifiers)):
        weak_classify_result=weak_classify(data_array,classifiers[i]['dim'],classifiers[i]['threshold'],classifiers[i]['sign'])
        aggre_classify_result+= classifiers[i]['alpha']*weak_classify_result
    return np.sign(aggre_classify_result)


def predict(data_array,lables,classifiers):
    lables=np.array(lables).reshape(len(lables),1)
    '''
    predict_result=[]
    for i in range(data_array.shape[0]):
        aggre_classify_result=adaboostclassify(data_array[i],classifiers)
        predict_result.append(aggre_classify_result)
    '''
    aggre_classify_result=adaboostclassify(data_array,classifiers)
    predict_result=np.array(aggre_classify_result).reshape(len(lables),1)
    predict_error=np.zeros((len(lables),1))
    predict_error[predict_result!=lables]=1
    predict_error_rate=predict_error.sum()/len(lables)
    return predict_result,predict_error_rate

