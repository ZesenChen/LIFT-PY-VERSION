import numpy as np
import sys
import evaluate as ev
import scipy.io as sio
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.preprocessing import scale

sys.path.append('F:\libsvm-3.22\python')

from svmutil import *

path = 'F:/clone/multi-label-dataset/sf dataset/'
dataset = 'yeast'
fold_num = 10

class LIFT(object):
    def __init__(self, sf_num):
        self.feat_num = sf_num
        self.svm = []
        self.binary = []
        
    def fit(self, train_data, train_target):
        #sp_features = []
        self.train_data = train_data
        self.train_target = train_target
        data_num,class_num = train_target.shape
        for i in range(class_num):
            begin = int(sum(self.feat_num[:i]))
            end = int(begin + self.feat_num[i])
            if sum(train_target[:,i])==0: 
                self.binary.append(0)
            elif sum(train_target[:,i])==data_num:
                self.binary.append(1)
            else: self.binary.append(2)
            prob = svm_problem(train_target[:,i].tolist(), \
                               train_data[:,begin:end].tolist())
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
            begin = sum(self.feat_num[:i])
            end = begin + self.feat_num[i]
            tmp_pre_labels,tmp,tmpoutputs = svm_predict(test_target[:,i].tolist(), \
                                                         test_data[:,begin:end].tolist(), \
                                                         self.svm[i],'-b 1')
            if self.binary[i] == 2:
                pre_labels[:,i] = tmp_pre_labels
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
    result = np.zeros((10,5))
    for i in range(fold_num):
        print('the ',i,'th fold.')
        filename = path+dataset+'/'+dataset+'_sf_'+str(i+1)+'_fold.mat'
        Set = sio.loadmat(filename)
        sf_num,train_data,train_target,test_data,test_target = \
        Set['sf_num'][0],Set['train_data'],Set['train_target'], \
        Set['test_data'],Set['test_target']
        train_target[train_target==-1]=0
        test_target[test_target==-1]=0
        #kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=2017)
        lift = LIFT(sf_num)
        lift.fit(train_data,train_target)
        tmp = lift.predict(test_data,test_target)
        result[i] = np.array(tmp)
    #print(result)
    mean = np.mean(result,0)
    std = np.std(result,0)
    print('Hamming Loss:',mean[0],'+-',std[0])
    print('Ranking Loss:',mean[1],'+-',std[1])
    print('Coverage:',mean[2],'+-',std[2])
    print('One Error:',mean[3],'+-',std[3])
    print('Average Precision:',mean[4],'+-',std[4])


