#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 09:26:40 2023

@author: stacy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 11:16:05 2023

"""

# import sklearn

from sklearn.decomposition import PCA
import os
import sys
import time

fatherpath = 'D:\Stacy'
sys.path.append(fatherpath)

from SDAE import stacked_denoising_autoencoder_node_embedding
import numpy as np
import pickle
from multiprocessing import Pool

import networkx as nx

from tools import *

data_set = ['Yeast', 'Human', 'Matine','Drosophila']
DataSetDir = data_set[1]


class MyClass(object):
    def __init__(self, PPI_Pos,PPI_Neg,PPSim, kfCV):

        self.PPI_Pos = PPI_Pos
        self.PPI_Neg = PPI_Neg
        self.kfCV = kfCV
        self.index_pos = get_kfold_index(len(PPI_Pos), rseed = 1388, n_splits=kfCV)
        self.index_neg = get_kfold_index(len(PPI_Neg), rseed = 1389, n_splits=kfCV)
        self.PPSim = PPSim
        self.boolNP = 0

    def setCV(self,cv):
        self.cv = cv
    
    def setPara(self, alpha, beta, t):    
        
        self.alpha = np.round( alpha,3)
        self.beta = np.round( beta,2)
        self.t = int(t)
        self.SavePath = fatherpath +'/' +str(int(self.kfCV)) + '_Fold_CV/' +  DataSetDir + '/Embedding/k_fold_' + str(self.cv + 1) + \
                    '/t_' + str(int(self.t)) + '/alp_' + str(self.alpha) + '_beta_' + str(self.beta) + '/'
    
        if not os.path.exists(self.SavePath):
            os.makedirs(self.SavePath)
        
        print('Current Path:',self.SavePath)
        

    def ConstructPPINetwork(self, ppis,index_list):
        NumSam = len(self.PPSim)
        TarAdj = np.zeros([NumSam, NumSam])
        TarAdj[ppis[index_list, 0],
               ppis[index_list, 1]] = 1
        TarAdj[ppis[index_list, 1],
               ppis[index_list, 0]] = 1
        return TarAdj

    def ConstructMatrixQ(self,PPI_Network):
        # A = gaussian_similarity_matrix(PPI_Network, gamma=1.0)
        A = gaussian_similarity(PPI_Network)
        R = self.PPSim
        num_raw, num_col = self.PPSim.shape
        Pt = np.eye(num_raw)
        Q = np.zeros([num_raw, num_col])
        #Cal：pt1 = alph * A +(1-alph) *R
        #Q = Q + np.power(beta,k) * Pt
        for k in range(int(self.t)):
            print('Processing Fusion Matrix layer t :'+str(k))
            Pt = self.alpha * np.dot(Pt , A) + (1-self.alpha) * R
            if k==0:
                Q = Pt
            else:
                Q = Q + np.power(self.beta,k+1) * \
                column_max_min_normalization(Pt)
        Q = column_max_min_normalization(Q)
        

        # np.save(tarFile, Q)
        return Q, A
    
    
    def learning_Rep(self):
        tarFile = self.SavePath + 'SAE_Embedding_pos_neg.npy'
        if os.path.exists(tarFile):
            Embedding_pos_neg = np.load(tarFile)
        else:
            stime = time.time()
            train_index_pos = np.array( np.where(self.index_pos != self.cv))
            PPI_Network_pos = self.ConstructPPINetwork(self.PPI_Pos,train_index_pos)
            Q_pos = self.ConstructMatrixQ(PPI_Network_pos)
            print('calculating NetSim costing time:', time.time() - stime)
            stime = time.time()
            tarQ = self.SavePath + 'Q_pos.npy'
            np.save(tarQ,Q_pos )
            train_index_neg = np.array( np.where(self.index_neg != self.cv))
            PPI_Network_neg = self.ConstructPPINetwork(self.PPI_Neg,train_index_neg)
            Q_neg = self.ConstructMatrixQ(PPI_Network_neg)
            tarQ = self.SavePath + 'Q_neg.npy'
            np.save(tarQ,Q_neg )
            print('calculating NetSim costing time:', time.time() - stime)
            stime = time.time()
            Embedding_pos_neg = self.Feature_Rep(
                np.hstack((Q_pos, Q_neg)))
            np.save(tarFile,Embedding_pos_neg )
            print('calculating SAE costing time:', time.time() - stime)
        return Embedding_pos_neg


    def Feature_Rep(self, FusionMatrix):
        
        Embedding, loss_value = stacked_denoising_autoencoder_node_embedding(
            FusionMatrix, output_dim = 128, encoder_params = [1024,512], learning_rate=0.001, batch_size=32, epochs=200, noise_factor=0.1)

        return Embedding


# if __name__ == '__main__':

FileProtein_pos = fatherpath + '/Data/' + DataSetDir + '/PPI_No_Pos.txt'
FileProtein_neg = fatherpath+ '/Data/' + DataSetDir + '/PPI_No_Neg.txt'
FileProteinFasta = fatherpath + '/Data/' + DataSetDir + '/Sequence.fa'
FileProteinSim = fatherpath + '/Data/' + DataSetDir+ '/Levenshtein_Sim.npy'

PPI_Pos = np.loadtxt(FileProtein_pos,dtype= np.int32,delimiter='\t')
PPI_Neg = np.loadtxt(FileProtein_neg,dtype= np.int32,delimiter='\t')
PPI_Pos = PPI_Pos - 1
PPI_Neg = PPI_Neg - 1

if os.path.exists(FileProteinSim):
    print('Loading Protein Similarity')
    PPSim = np.load(FileProteinSim)
    tarFile = fatherpath + '/Data/' + DataSetDir+ '/Levenshtein_Sim.csv'
    np2txt(PPSim,tarFile)
else:
    print('calulating Protein Similarity')
    PPSim = compute_Leven(GetProteinInfo( FileProteinFasta))
    np.save(FileProteinSim, PPSim)

kfCV = 5
ob = MyClass( PPI_Pos,PPI_Neg,PPSim, kfCV = kfCV)

alp_list = np.linspace(0, 1, 9)
# beta_list = np.linspace(0.7, 1, 4)
# beta_list = np.linspace(1, 10, 9)
# t_list = np.linspace(1, 7, 4).astype(int)
# alp_list = [0.625]
beta_list = [1]
t_list = [1]
accs = np.zeros([kfCV,len(alp_list)])

scores= np.zeros([kfCV,6])
pred_list = []
test_list =[]
fpr_list =[]
tpr_list = []
for cv in range(kfCV):#kfCV
# for cv in range(kfCV-1,-1,-1):
    ob.setCV(cv)
    for i in range(len(alp_list)):
        for j in range(len(beta_list)):
            for k in range(len(t_list)):
                alpha = alp_list[i]
                beta = beta_list[j]
                t = t_list[k]
                ob.setPara( alpha, beta, t)
                
                SAE_Embedding = ob.learning_Rep()
                
                Train_Feature, Test_Feature, Train_labels, Test_labels = \
                    GenFeatureSet(SAE_Embedding, ob.PPI_Pos, ob.PPI_Neg,\
                              ob.index_pos, ob.index_neg, ob.cv)
                

                tarFile = ob.SavePath + 'SAE_SVM_best_params.npy'
                tarSVMpkl = ob.SavePath + "SAE_SVM_best_params.pkl"
                accs[cv][i], _ , y_pred, y_probs =go_optSVM(Train_Feature, Test_Feature, Train_labels, Test_labels, tarFile,tarSVMpkl)
                
                
                # tarFile = ob.SavePath + 'SAE_XGB_best_params.npy'
                # tarSVMpkl = ob.SavePath + "SAE_XGB_best_params.pkl"
                # _, _, y_pred, y_probs= go_optXGB(Train_Feature, Test_Feature, Train_labels, Test_labels, tarFile,tarSVMpkl)
                
                # accs[cv][i], y_pred, y_probs = go_NB(Train_Feature, Test_Feature, Train_labels, Test_labels)
                

                y_pred = (y_probs >= 0.5).astype(int)
                scores[cv,0], scores[cv,1],scores[cv,2],scores[cv,3],scores[cv,4], scores[cv,5], fpr, tpr = calculate_metrics_and_roc(Test_labels, y_pred, y_probs)
                
                pred_list.append(y_pred)
                test_list.append(Test_labels)
                fpr_list.append(fpr)
                tpr_list.append(tpr)

# scores[:, 0:3] = np.round(scores[:, 0:3]*100, 2)

# avg = np.zeros([1,9])
# for i in range(len(alp_list)):
#     avg[0][i] = np.mean(scores[:][i])
