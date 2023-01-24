import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import joblib
import json
import glob
import logging
from collections import defaultdict
import pickle

date_var = 'Date'
store_var = 'Store'
target = 'Weekly_Sales'
term = 'week'

#전처리 class
class Preprocessing:
    def __init__(self, log_name, data, var_list, num_var, obj_var, target, date_var, store_var, unit, anomaly_per=10):
        self.df = data                                     # 데이터
        self.var_list = var_list                           # 전체 변수 리스트
        self.num_var = num_var                             # 수치형 변수 리스트
        self.obj_var = obj_var                             # 문자형 변수 리스트
        self.target = target                               # 타겟 변수
        self.date_var = date_var                           # 시간 변수
        self.store_var = store_var                         # 지점(상품 변수)
        self.unit = unit                                   # 시간 단위
        
        self._anomaly_ratio = int(anomaly_per)             # 지정 결측 범위
        self._anomaly_percentage = int(anomaly_per) / 100  # 지정 결측 범위
        
        self.logger = logging.getLogger(log_name)
        
        #결측치 처리 먼저 진행
        self.df = self.na_preprocess(self.df, self._anomaly_ratio)
        
        self.df, self.tds_df = self.tds_preprocess(self.df, self.target, self.date_var, self.store_var)
        
        # 표준화
        self.df = self.standardize(self.df, self.num_var)
        
        # 라벨 인코딩
        self.df = self.label_encoder(self.df, self.obj_var)
        
        # 일반 전처리 완료
        self.df = self.get_df()
        
        # 시계열 전처리 수행
        self.df = self.ts_preprocess(self.df, self.target, self.date_var, self.store_var, self.unit)
    
    
        # 결측치 확인 및 처리
    def na_preprocess(self, df, anomaly_per):
        
        self.logger.info('결측치 처리')
        
        try:
            #Column별 결측치 n% 이상 있을 경우 제외
            remove_v1 = round(df.isnull().sum() / len(df)*100, 2)
            tmp_df = df[remove_v1[remove_v1 < anomaly_per].index]
        
            #Row별 결측치 n% 이상 있을 경우 제외
            idx1 = len(tmp_df.columns) * 0.7
        
        except:
            self.logger.exception('결측치 처리에 문제가 발생하였습니다')
        
        self.logger.info(f'결측치 처리 이후 데이터 구성: {df.shape[0]} 행, {df.shape[1]}열')                  
        
        return tmp_df.dropna(thresh=idx1, axis=0)
    
  
    def tds_preprocess(self, df, target, date_var, store_var):
        #target, date, store
        self.logger.info('전처리를 위한 target, date, store 분리')
        
        #식별 변수가 있을 수도 있고 없을 수도 있다(0119)
        
        try:
            tds_df = df[[target, date_var, store_var]]
            df = df.drop([target, date_var, store_var], axis=1)
            
            if date_var in self.num_var:
                self.num_var.remove(date_var)    
            else : self.obj_var.remove(date_var)

            if store_var in self.num_var:
                self.num_var.remove(store_var)    
            else : self.obj_var.remove(store_var)

            if target in self.num_var:
                self.num_var.remove(target)    
            else : self.obj_var.remove(target)
            
            
        except:
            self.logger.exception(' target, date, store 분리 처리에 문제가 발생하였습니다')
            
        return df, tds_df
        
    
#     # 이상치 제거 절차 삭제(230119)
        
    
    #정규화
    def standardize(self, df, num_var):
                                  
        self.logger.info('정규화 진행')
        try:        
            if num_var:
                num_data = df.loc[:, num_var]
                non_num_data = df.drop(set(num_var), axis=1)

                #표준화
                std_scaler = StandardScaler()
                fitted = std_scaler.fit(num_data)
                output = std_scaler.transform(num_data)
                num_data = pd.DataFrame(output, columns = num_data.columns, index=list(num_data.index.values))

                tmp_df = pd.concat([non_num_data, num_data], axis=1)
            else:
                tmp_df = df
        except:
            self.logger.exception('정규화 진행 중에 문제가 발생하였습니다')                                      
                                  
        return tmp_df
        
    
    #문자형 변수를 수치형으로 변환
    def label_encoder(self, df, obj_var):
                                  
        self.logger.info('라벨 인코딩 진행')
        try:                              
            if obj_var:
                obj_data = df.loc[:, obj_var]
                non_obj_data = df.drop(set(obj_var), axis=1)

                #인코딩
                lbl_en = LabelEncoder()
                lbl_en = defaultdict(LabelEncoder)
                obj_data = obj_data.apply(lambda x:lbl_en[x.name].fit_transform(x))
            
                #라벨 인코딩 저장    
                pickle.dump(lbl_en, open('storage/label_encoder.sav', 'wb'))
                
            
                tmp_df = pd.concat([obj_data, non_obj_data], axis=1)
                
            else:
                tmp_df = df
                                  
        except:
            self.logger.exception('수치형 변환 중에 문제가 발생하였습니다')                                      
                                 
        return tmp_df
    
    
    def get_df(self):
        
        self.df = pd.concat([self.df, self.tds_df], axis=1)

        self.logger.info('전처리 완료')
        self.logger.info('\n')
        self.logger.info(self.df.head())
        
        
        return self.df

    
    def ts_preprocess(self, data, target, date_var, store_var, unit):
        
        self.logger.info('시계열용 전처리 진행')
        try:    
            #store_var type이 str이어야 함
            data[store_var] = data[store_var].astype(str)
            data[date_var] = pd.to_datetime(data[date_var], dayfirst=True)
            print(data.dtypes)
            df = pd.DataFrame()
            for store in data[store_var].unique():
                tmp_df = data.loc[data[store_var] == store,:].sort_values(date_var).reset_index(drop=True)
                tmp_df['time_idx'] = tmp_df.index
                df = pd.concat([df, tmp_df], axis=0)

            df.reset_index(drop=True, inplace=True)

            print(df.head())
            # add additional features
            df[unit] = df[date_var].dt.week.astype(str).astype("category")  # categories have be strings
            df[f"log_{target}"] = np.log(df[target] + 1e-8)
            df[f"avg_{target}_by_{store_var}"] = df.groupby(["time_idx", store_var], observed=True)[target].transform("mean")
            
        except:
            self.logger.exception('시계열용 전처리 진행 중에 문제가 발생하였습니다')     

        return df
