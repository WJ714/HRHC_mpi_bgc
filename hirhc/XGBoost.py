# -*- coding: utf-8 -*-
"""
@author: mjung
"""

import numpy as np
import copy
import xgboost as xgb
import xarray as xr


mfqc=0.8
minfracOKdata=0.3

nfolds=3

params = {
  'colsample_bynode': 1,
  'learning_rate': 0.05,
  'max_depth': 6,
  'num_parallel_tree': 1,
  'objective': 'reg:squarederror',
  'subsample': 0.5,
  'tree_method': 'auto',
  'min_child_weight': 5,  
  }



class xgbTrain(object):
    def __init__(self, X,Y,idxTrain=None,idxTest=None,idxPred=None,idxFI=None,chrom=None,x_mono=None,interactions=None,xgbparams=None,ntrees=1000,early_stopping_rounds=20,trainModel=True,calcFI=False,retrainWithTest=True):
        
        #idxTrain, idxTest, idxPred: indices for training (boleans), test, prediction
        #idxTest: used to prevent overfitting
        #chrom: bolean for features to include (True=include)
        
        sz=np.shape(X)
        n=sz[0]
        nV=sz[1]
        if xgbparams is None:    
            xgbparams = {
              'colsample_bynode': 1,
              'learning_rate': 0.05,
              'max_depth': 10,
              'num_parallel_tree': 1,
              'objective': 'reg:squarederror',
              'subsample': 0.5,
              'tree_method': 'auto',
              'min_child_weight': 5
              }

        if chrom is None:
            chrom=np.ones((1,nV),dtype=bool)
        if idxTrain is None:
            idxTrain=np.ones(n,dtype=bool)
            #retrainWithTest=False
        if idxFI is None:
            idxFI=np.ones(n,dtype=bool)
            
        if x_mono is None:
            x_mono=np.zeros(nV,dtype=int)
        else:
            xgbparams['monotone_constraints']=getMonoConstrains4chrom(x_mono,chrom)
            
            
        if interactions is not None:
            intStr=getInteractions4chrom(interactions,chrom)

            xgbparams['interaction_constraints'] = intStr
            
        self.szX=sz
        self.xgbparams=xgbparams
        self.chrom=chrom
        self.X=X
        self.Y=Y
        self.idxTrain=idxTrain
        self.idxTest=idxTest
        self.ntrees=ntrees
        self.early_stopping_rounds=early_stopping_rounds
        self.idxFI=idxFI
        self.idxPred=idxPred
        self.retrainWithTest=retrainWithTest
        
        if trainModel:
            bst=self.train()
            
        if calcFI:            
            fi=self.getFI()
            
    def train(self):
        
        vidx=np.where(np.squeeze(self.chrom))[0]
  
        Dtrain=xgb.DMatrix(self.X[:,vidx][self.idxTrain,:], label=self.Y[self.idxTrain])
        if self.idxTest is not None:                
        
            Dpred=xgb.DMatrix(self.X[:,vidx][self.idxTest,:], label=self.Y[self.idxTest])            
            evallist=[(Dpred,'eval')]    
            bst = xgb.train(self.xgbparams,Dtrain, self.ntrees,evallist,early_stopping_rounds=self.early_stopping_rounds,verbose_eval=False)
            best_ntree=bst.best_iteration # due to the update of 'best_ntree_limit' to 'best_iteration'
            if best_ntree == self.ntrees:
                print('Warning: underfitting!, Increase LR or nTrees')
            mse=bst.best_score**2
            self.mse=mse
            self.mef=1-mse/np.mean((self.Y[self.idxTest]-np.mean(self.Y[self.idxTest]))**2)
            self.best_ntree=best_ntree   
            if self.retrainWithTest:
                valTr=np.logical_or(self.idxTrain,self.idxTest)
                Dtrain=xgb.DMatrix(self.X[:,vidx][valTr,:], label=self.Y[valTr])
                bst = xgb.train(self.xgbparams,Dtrain, best_ntree)
             
        else:
            bst = xgb.train(self.xgbparams,Dtrain, self.ntrees)
        self.model=bst                

        #if self.idxFI is not None:
        
        return(bst)
        
    
    def pred(self,Xpred=None):
        vidx=np.where(np.squeeze(self.chrom))[0]
        if Xpred is None:
            Dpred=xgb.DMatrix(self.X[:,vidx][self.idxPred,:])   
        else:
            Dpred=xgb.DMatrix(Xpred[:,vidx])   
        y_pred=self.model.predict(Dpred)
        self.ypred=y_pred
        return(y_pred)

    

def getMonoConstrains4chrom(x_mono,chrom):
    vidx=np.where(np.squeeze(chrom))[0]
    cX_mono=x_mono[vidx]
    tmp='('
    for s in np.arange(np.size(cX_mono)):
        tmp=tmp+str(int(cX_mono[s]))+','

    cX_mono_str=tmp[0:-1]+')'
    return(cX_mono_str)    
    
        
def getInteractions4chrom(interactions,chrom):
    vidx=np.where(np.squeeze(chrom))[0]
    new_idx=np.arange(np.size(vidx))
    ngroups=len(interactions)           
    interactions_chrom=copy.copy(interactions)
    for s in np.arange(ngroups):
        cgroup=interactions[s]
        tmp=np.intersect1d(vidx,cgroup,return_indices=True)
        cgroup_new=new_idx[tmp[1]]
        interactions_chrom[s]=cgroup_new
    return(interactionIdx2interactionString(interactions_chrom))    
        
def interactionIdx2interactionString(_interactions):
    
    #params['interaction_constraints'] = '[[0, 2], [1, 3, 4], [5, 6]]'
    ngroups=len(_interactions)           
    tmp='['
    for s in np.arange(ngroups):
        #tmp=tmp+'['
        cgroup=_interactions[s]
        tmp=tmp+str(list(cgroup))
        if s < (ngroups-1):
            tmp=tmp+','
    tmp=tmp+']'
    return(tmp)
