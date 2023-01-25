import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
import joblib
import json
import glob
import logging


class Modeling:
    def __init__(self, df, target_var, date_var, store_list, n):
        self.df = df                                           # 데이터
        self.target_var = target_var                           # 타겟 변수
        self.date_var = date_var                               # 시간 변수
        self.store_list = store_list                           # 지점(상품 변수)
        self.unit = unit                                       # 시간 단위
        self.predict_n = n                                     # 예측 기간
        
        
        self.logger = logging.getLogger(log_name)
       
        
        self.df = self.id_modeling(self.df, self.unique_id)
        self.X_train, self.X_test, self.y_train, self.y_test = self.train_test_split(self.df, self.target, self.over_sampling)
            
        self.cat_idxs, self.cat_dims = self.get_tabnet_param()
        self.tab_fit_predict(self.cat_idxs, self.cat_dims, self.X_train, self.y_train, self.X_test, self.y_test)
        self.lgb_fit_predict(self.X_train, self.y_train, self.X_test, self.y_test, self.hpo)
        
        self.best_model_name, self.best_model, self.best_test = self.get_best_model()
        
        self.get_eval(self.best_model, self.best_test)
        self.get_plot(self.best_model, self.best_model_name, self.X_train)
            
    def id_modeling(self, df, unique_id):
        
        self.logger.info('모델링을 위한 id 분리')
        try:
            id_df = df[[unique_id]]
            df = df.drop([unique_id], axis=1)
            
        except:
            self.logger.exception('모델링을 위한 id분리에 문제가 발생하였습니다')
        
        return df
    
    #train, valid 분리
    def train_test_split(self, df, target, over_sampling):
        
        self.logger.info('train valid 분리')
