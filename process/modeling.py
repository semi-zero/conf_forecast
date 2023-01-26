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

#fbprophet
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
import optuna
from optuna.samplers import TPESampler



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
       
        
        self.val_df = self.fb_fit_predict(self.df, self.target_var, self.date_var, self.store_list, self.predict_n)
            
    def fb_fit_predict(self, df, target_var, date_var, store_list, predict_n):
        
        self.logger.info('fbprophet 데이터 준비')
    
        #store_list가 하나일 때
        if len(store_list) == 1:
            val_df = pd.DataFrame()
            store_var = store_list[0]
            for store in df[store_var].unique():
                #
                fb_df = df.loc[df[store_var] == store,:]
                fb_df['ds'] = fb_df[date_var]
                fb_df['y'] = fb_df[target_var]        
                fb_df['cap'] = np.max(fb_df[target_var].values)
                fb_df['floor'] = np.min(fb_df[target_var].values)
                #fb_df = fb_df[['ds','y','cap','floor']]
                
                predict_size = predict_n
                fb_train = fb_df.iloc[:-predict_size, :]
                fb_var = fb_df.iloc[-predict_size:, :]
                
                def objective(trial):
                    params = {
                    'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.005, 5),
                    'changepoint_range': trial.suggest_float('changepoint_range', 0.8, 0.9),
                    'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.1, 10),
                    'holidays_prior_scale': trial.suggest_float('holidays_prior_scale', 0.1, 10),
                    'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['multiplicative', 'additive']),
                    'growth': trial.suggest_categorical('growth', ['linear', 'logistic']),
                    'weekly_seasonality': trial.suggest_int('weekly_seasonality', 5, 10),
                    'yearly_seasonality': trial.suggest_int('yearly_seasonality', 1, 20)
                    }

                    m = Prophet(**params)
                    m.fit(fb_train[['y','ds','cap','floor']])
                    preds = m.predict(fb_var[['ds','cap','floor']])

                    mae_score = mean_absolute_error(fb_var['y'], preds['yhat'])

                    return mae_score
                
                sampler = TPESampler(seed=42)
                study = optuna.create_study(direction='minimize', sampler=sampler,)
                study.optimize(objective, n_trials = 1)
                
                params = study.best_params
                
                m = Prophet(**params)
                m.fit(fb_train[['y','ds','cap','floor']])
                val_preds = m.predict(fb_var[['ds','cap','floor']])
                val_preds = val_preds[['ds','yhat']]
                print(fb_var.head())
                val_real = fb_var[['y',store_var, date_var]]
                val_preds_df = pd.merge(val_preds, val_real, left_on='ds', right_on=date_var, how='inner')
                
                val_df = pd.concat([val_df, val_preds_df], axis=0) 
                
            val_df.to_csv('val_df.csv', index=False)
                
            return val_df


                