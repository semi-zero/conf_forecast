import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
import joblib
import json
import glob
import logging
from datetime import timedelta

#fbprophet
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
import optuna
from optuna.samplers import TPESampler

from . import hpo


class Modeling:
    
    def __init__(self, log_name, df, target_var, date_var, store_list, unit, predict_n, HPO):
        self.df = df                                           # 데이터
        self.target_var = target_var                           # 타겟 변수
        self.date_var = date_var                               # 시간 변수
        self.store_list = store_list                           # 지점(상품 변수)
        self.unit = unit                                       # 시간 단위
        self.predict_n = predict_n                             # 예측 기간
        self.HPO = HPO                                         # 하이퍼파라미터 튜닝 여부
        
        
        self.logger = logging.getLogger(log_name)
       
        
        self.val_df = self.fb_fit_predict(self.df, self.target_var, self.date_var, self.store_list, self.predict_n, self.HPO)
            
    def fb_fit_predict(self, df, target_var, date_var, store_list, predict_n, HPO):
        
        self.logger.info('fbprophet 데이터 준비')
        print(store_list)
        #store_list가 하나일 때
        if len(store_list) == 1:
            val_df = pd.DataFrame()
            pred_df = pd.DataFrame()
            store_var = store_list[0]
            for store in df[store_var].unique():
                fb_df = df.loc[df[store_var] == store,:]
                fb_df['ds'] = fb_df[date_var]
                fb_df['y'] = fb_df[target_var]        
                fb_df['cap'] = np.max(fb_df[target_var].values)
                fb_df['floor'] = np.min(fb_df[target_var].values)
                #fb_df = fb_df[['ds','y','cap','floor']]
                
                predict_size = predict_n
                fb_train = fb_df.iloc[:-predict_size, :]
                fb_var = fb_df.iloc[-predict_size:, :]
                
                
                if HPO :
                    self.logger.info('fb HPO 진행') 
                    parameters = hpo.HyperOptimization(train = fb_train, valid = fb_var, model = 'fb').best_params
                    self.logger.info(f'fb HPO 진행 후 parameters: {parameters}')
                else:
                    parameters = {'changepoint_prior_scale': 1.8, 'changepoint_range': 0.8, 'seasonality_prior_scale': 7.3, 'holidays_prior_scale': 6, 'seasonality_mode': 'multiplicative', 'weekly_seasonality': 5, 'yearly_seasonality': 18}
                    
                #validate 후 validate_df 생성
                m = Prophet(**parameters)
                m.fit(fb_train[['y','ds','cap','floor']])
                val_preds = m.predict(fb_var[['ds','cap','floor']])
                val_preds = val_preds[['ds','yhat']]
                val_real = fb_var[['y', date_var]]
                val_preds_df = pd.merge(val_preds, val_real, left_on='ds', right_on=date_var, how='inner')
                
                val_df = pd.concat([val_df, val_preds_df], axis=0) 
                val_df[store_var] = store
                #predicat_date 생성 후 예측 predict_df생성
                
                #m.fit(fb_df[['ds','cap','floor']])
                last_date = fb_df[date_var].iloc[-1:].tolist()[0]
                predict_date = [last_date + timedelta(weeks=i) for i in range(1, predict_n+1)] #weeks, days 변경 가능
                test_df = pd.DataFrame({'ds': predict_date})
                test_df['cap'] = fb_df['cap'].values[0]
                test_df['floor'] = fb_df['floor'].values[0]

                preds = m.predict(test_df[['ds','cap','floor']])
                preds = preds[['ds','yhat']]
                preds[store_var] = store
                pred_df = pd.concat([pred_df, preds], axis=0)

                
            val_df.to_csv('val_df.csv', index=False)
            pred_df.to_csv('pred_df.csv', index=False)
                
            return val_df
        
        elif len(store_list) == 2:
            print(2)
            val_df = pd.DataFrame()
            pred_df = pd.DataFrame()
            
            for store_var_0, store_var_1 in df.drop_duplicates(store_list)[store_list].values:
                fb_df = df.loc[(df[store_list[0]]==store_var_0)&(df[store_list[1]]==store_var_1), :]               
                fb_df['ds'] = fb_df[date_var]
                fb_df['y'] = fb_df[target_var]        
                fb_df['cap'] = np.max(fb_df[target_var].values)
                fb_df['floor'] = np.min(fb_df[target_var].values)

                predict_size = predict_n
                fb_train = fb_df.iloc[:-predict_size, :]
                fb_var = fb_df.iloc[-predict_size:, :]

                print(HPO)
                if HPO :
                    self.logger.info('fb HPO 진행') 
                    parameters = hpo.HyperOptimization(train = fb_train, valid = fb_var, model = 'fb').best_params
                    self.logger.info(f'fb HPO 진행 후 parameters: {parameters}')
                
                else:
                    parameters = {'changepoint_prior_scale': 1.8, 'changepoint_range': 0.8, 'seasonality_prior_scale': 7.3, 'holidays_prior_scale': 6, 'seasonality_mode': 'multiplicative', 'weekly_seasonality': 5, 'yearly_seasonality': 18}

                #validate 후 validate_df 생성
                m = Prophet(**parameters)
                m.fit(fb_train[['y','ds','cap','floor']], algorithm='Newton')
                val_preds = m.predict(fb_var[['ds','cap','floor']])
                val_preds = val_preds[['ds','yhat']]
                val_real = fb_var[['y', date_var]]
                val_preds_df = pd.merge(val_preds, val_real, left_on='ds', right_on=date_var, how='inner')

                val_df = pd.concat([val_df, val_preds_df], axis=0) 
                val_df[store_list[0]] = store_var_0
                val_df[store_list[1]] = store_var_1
                #predicat_date 생성 후 예측 predict_df생성

                #m.fit(fb_df[['ds','cap','floor']])
                last_date = fb_df[date_var].iloc[-1:].tolist()[0]
                predict_date = [last_date + timedelta(days=30*i) for i in range(1, predict_n+1)] #weeks, days 변경 가능
                test_df = pd.DataFrame({'ds': predict_date})
                test_df['cap'] = fb_df['cap'].values[0]
                test_df['floor'] = fb_df['floor'].values[0]

                preds = m.predict(test_df[['ds','cap','floor']])
                preds = preds[['ds','yhat']]
                preds[store_list[0]] = store_var_0
                preds[store_list[1]] = store_var_1
                pred_df = pd.concat([pred_df, preds], axis=0)


            val_df.to_csv('val_df.csv', index=False)
            pred_df.to_csv('pred_df.csv', index=False)

            return val_df


                