import numpy as np
import sys
import evaluate as ev
import scipy.io as sio
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.feature_selection import SelectFromModel,RFE,RFECV
from sklearn.linear_model import Ridge,RidgeCV,Lasso,LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.preprocessing import scale
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

sys.path.append('F:\libsvm-3.22\python')

from svmutil import *

path = 'F:/clone/LIFT-PY-VERSION/specific features dataset/'
dataset = 'recreation_sfeatures_all.mat'

class EM_LIFT(object):
    def __init__(self, sf_num):
        self.feat_num = sf_num
        self.svm = []
        self.binary = []
        self.feat_sel_model = []
        
    def fit(self, train_data, train_target):
        #sp_features = []
        self.train_data = train_data
        self.train_target = train_target
        data_num,class_num = train_target.shape
        for i in range(class_num):
            #begin = int(sum(self.feat_num[:i]))
            #end = int(begin + self.feat_num[i])
            if sum(train_target[:,i])==0: 
                self.binary.append(0)
                self.feat_sel_model.append([])
                self.svm.append([])
            elif sum(train_target[:,i])==data_num:
                self.binary.append(1)
                self.feat_sel_model.append([])
                self.svm.append([])
            else:
                self.binary.append(2)
                svc = LinearDiscriminantAnalysis()
                svc.fit(train_data,train_target[:,i])
                #rfe = SelectFromModel(svc)
                #rlasso = rlogr.fit(train_data,train_target[:,i])
                #model = SelectFromModel(rlogr)
                #rfe.fit(train_data,train_target[:,i])
                self.feat_sel_model.append(svc)
                new_train_data = svc.transform(train_data)
                prob = svm_problem(train_target[:,i].tolist(), \
                                   new_train_data.tolist())
                param = svm_parameter()
                param.kernel_type = LINEAR
                param.C = 1
                param.probability = 1
                self.svm.append(svm_train(prob,param))

    def predict(self, test_data, test_target):
        data_num,class_num = test_target.shape
        pre_labels = np.zeros((data_num,class_num))
        outputs = np.zeros((data_num,class_num))
        for i in range(class_num):
            if self.binary[i] == 2:
                new_test_data = self.feat_sel_model[i].transform(test_data)
                pre_labels[:,i],tmp,tmpoutputs = svm_predict(test_target[:,i].tolist(), \
                                                         new_test_data.tolist(), \
                                                         self.svm[i],'-b 1')
                pos_index = 0 if self.svm[i].label[0]==1 else 1
                outputs[:,i] = np.array(tmpoutputs)[:,pos_index]
            else:
                pre_labels[:,i] = self.binary[i]
                outputs[:,i] = self.binary[i]
            
        return [ev.HammingLoss(pre_labels,test_target),
                ev.rloss(outputs,test_target),
                ev.Coverage(outputs,test_target),
                ev.OneError(outputs,test_target),
                ev.avgprec(outputs,test_target)]
    
if __name__ == '__main__':
    Set = sio.loadmat(path+dataset)
    sf_num,data,target = Set['sf_num'][0],scale(Set['data']),Set['target']
    target[target==-1]=0
    kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=2017)
    emlift = EM_LIFT(sf_num)
    result = [0]*5
    j = 0;
    for dev_index, val_index in kf.split(data):
        j = j+1
        print('the',j,'th circle.')
        dev_X, val_X = data[dev_index,:],data[val_index,:]
        dev_y, val_y = target[dev_index,:],target[val_index,:]
        emlift.fit(dev_X,dev_y)
        tmp = emlift.predict(val_X,val_y)
        result = [result[i]+tmp[i] for i in range(5)]
    #print(result)
    print('Hamming Loss:',result[0]/10)
    print('Ranking Loss:',result[1]/10)
    print('Coverage:',result[2]/10)
    print('One Error:',result[3]/10)
    print('Average Precision:',result[4]/10)


