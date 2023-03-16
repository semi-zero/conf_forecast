import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import joblib
import json
import glob
import logging
from datetime import timedelta
import shutil
import datetime

#ARIMA
import statsmodels.api as sm
from datetime import timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima
from statsmodels.tsa.stattools import adfuller

#ETS
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import Holt

#prophet
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
import optuna
from optuna.samplers import TPESampler

#Temporal Fusion Transformer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

from . import hpo

def mape(y_test, y_pred):
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100 

def smape(y_test, y_pred):
    v = 2 * np.abs(y_pred-y_test) / (np.abs(y_pred) + np.abs(y_test))
    output = np.mean(v) * 100
    return output
        
class Modeling:
    
    def __init__(self, log_name, df, target_var, date_var, store_list, unit, predict_n, HPO, model_type='auto'):
        self.df = df                                           # 데이터
        self.target_var = target_var                           # 타겟 변수
        self.date_var = date_var                               # 시간 변수
        self.store_list = store_list                           # 지점(상품 변수)
        self.unit = unit                                       # 시간 단위
        self.predict_n = predict_n                             # 예측 기간
        self.HPO = HPO                                         # 하이퍼파라미터 튜닝 여부
        
        self.model_type = model_type
        self.model = dict()
        self.score = dict()
        self.result_df = dict()
        self.vi = dict()
        
        
        self.logger = logging.getLogger(log_name)
        
        self.start_time = datetime.datetime.now()
       
        self.score['MAPE']    = dict() #mape(y_test, y_pred)
        self.score['MAE']     = dict() #mean_absolute_error(y_test, y_pred)
        self.score['MSE']     = dict() #mean_squared_error(y_test, y_pred)
        self.score['RMSE']    = dict() #np.sqrt(mean_squared_error(y_test, y_pred))
        self.score['상관계수'] = dict() #r2_score(y_test, y_pred)
        self.score['정확성']   = dict() #1-MAPE
        
        self.result_df['train_df'] = dict()
        self.result_df['val_df'] = dict()
        self.result_df['pred_df'] = dict()
        
        
        
        #모델링 딕셔너리
        model_type_dict = {'ari' : self.ari_process()
                           ,'ets' : self.ets_process()
                           ,'fb'  : self.fb_process()
                           ,'tft'  : self.tft_process()}
        
        self.best_model_name, self.best_model = self.get_best_model()
        
        #학습 결과 화면을 위한 함수들
        #1. 분석 리포트
        self.report = self.make_report(self.target_var, self.date_var, self.unit, self.model_type, self.HPO, self.start_time)
        
        #2. 학습 결과 비교 화면
        #3. 변수 중요도와 시간 중요도
        self.test_score, self.valid_score, self.best_train_df, self.best_val_df, self.best_pred_df, self.vi_time, self.vi_static, self.vi_encoder =  self.get_eval(self.best_model_name)
    #################### ARI START####################
    
    def ari_process(self):
        if (self.model_type == 'ari') or (self.model_type == 'auto'): 
            self.ari_fit_predict(self.df, self.target_var, self.date_var, self.store_list, self.unit, self.predict_n, self.HPO)
        
    
    def ari_fit_predict(self, df, target_var, date_var, store_list, unit, predict_n, HPO):
        
        self.logger.info('arima 데이터 준비')
        
        
        if len(store_list) == 1 :
            store_list = ['dummy'] + store_list
            df.loc[:, 'dummy'] = 'dummy'
        
        train_df = pd.DataFrame()
        val_df = pd.DataFrame()
        pred_df = pd.DataFrame()
        
        self.logger.info('arima 모델링 시작')
        try:
            for store_var_0, store_var_1 in df.drop_duplicates(store_list)[store_list].values:
                ari_df = df.loc[(df[store_list[0]]==store_var_0)&(df[store_list[1]]==store_var_1), :]               
                ari_df.loc[:, 'ds'] = ari_df[date_var]
                ari_df.loc[:, 'y'] = ari_df[target_var]        

                predict_size = predict_n
                ari_train = ari_df.iloc[:-predict_size, :]
                ari_var = ari_df.iloc[-predict_size:, :]

                #auto arima best parameter pick
                ari = auto_arima(y = ari_train['y'].values, d = 1, start_p = 0, max_p = 3, start_q = 0 
                                  , max_q = 3, m = 1, seasonal = False , stepwise = True, trace=True)

                #arima fit 후 ari_var에 예측값 대입
                ari.fit(ari_train['y'].values)
                val_preds = ari.predict(n_periods = predict_size)
                ari_var.loc[:, 'pred'] = val_preds

                #train
                train= ari_train[[date_var, target_var]]
                train.loc[:, store_list[0]] = store_var_0
                train.loc[:, store_list[1]] = store_var_1

                train_df = pd.concat([train_df, train], axis=0)

                #valid
                val_preds_df = ari_var[[date_var, target_var, 'pred']]
                val_preds_df.loc[:, store_list[0]] = store_var_0
                val_preds_df.loc[:, store_list[1]] = store_var_1

                val_df = pd.concat([val_df, val_preds_df], axis=0)

                #predict_date 생성 후 예측 predict_df생성
                last_date = ari_df[date_var].iloc[-1:].tolist()[0]

                if unit =='day':
                    predict_date = [last_date + timedelta(days=i) for i in range(1, predict_n+1)] #weeks, days 변경 가능
                elif unit == 'week':
                    predict_date = [last_date + timedelta(days=7*i) for i in range(1, predict_n+1)] #weeks, days 변경 가능
                elif unit == 'month':
                    predict_date = [last_date + timedelta(days=30*i) for i in range(1, predict_n+1)] #weeks, days 변경 가능

                #test_df = pd.DataFrame({'ds':predict_date})
                test_df = pd.DataFrame({date_var : predict_date})
                ari.fit(ari_df['y'].values)
                preds = ari.predict(n_periods = predict_size)
                test_df.loc[:, 'pred'] = preds
                test_df.loc[:, store_list[0]] = store_var_0
                test_df.loc[:, store_list[1]] = store_var_1

                pred_df = pd.concat([pred_df, test_df], axis=0)
                
        except:
                self.logger.exception('arima 모델링 도중 문제가 발생하였습니다.')
                
        self.result_df['train_df']['ari'] = train_df
        self.result_df['val_df']['ari'] = val_df
        self.result_df['pred_df']['ari'] = pred_df
#         train_df.to_csv('result/ari_train_df.csv', index=False)
#         val_df.to_csv('result/ari_val_df.csv', index=False)
#         pred_df.to_csv('result/ari_pred_df.csv', index=False)
        
        
        #모델 평가
        y_test = val_df[target_var].values
        y_pred = val_df['pred'].values
        
        self.model['ari'] = ari
        
        self.score['MAPE']['ari']     = np.round(mape(y_test, y_pred), 3)
        self.score['MAE']['ari']      = np.round(mean_absolute_error(y_test, y_pred), 3)
        self.score['MSE']['ari']      = np.round(mean_squared_error(y_test, y_pred), 3)
        self.score['RMSE']['ari']     = np.round(np.sqrt(mean_squared_error(y_test, y_pred)), 3)
        self.score['상관계수']['ari']  = np.round(r2_score(y_test, y_pred), 3)
        self.score['정확성']['ari']    = np.round(100-mape(y_test, y_pred), 3)

        


    #################### ARI FINISH ####################
        
    #################### ETS START####################
    
    def ets_process(self):
        if (self.model_type == 'ets') or (self.model_type == 'auto'): 
            self.ets_fit_predict(self.df, self.target_var, self.date_var, self.store_list, self.unit, self.predict_n, self.HPO)
        
    
    def ets_fit_predict(self, df, target_var, date_var, store_list, unit, predict_n, HPO):
        
        self.logger.info('ets 데이터 준비')
        
        
        if len(store_list) == 1:
            store_list = ['dummy'] + store_list
            df.loc[:,'dummy'] = 'dummy'

        train_df = pd.DataFrame()
        val_df = pd.DataFrame()
        pred_df = pd.DataFrame()

        try:
            for store_var_0, store_var_1 in df.drop_duplicates(store_list)[store_list].values:
                ets_df = df.loc[(df[store_list[0]]==store_var_0)&(df[store_list[1]]==store_var_1), :]               
                ets_df.loc[:, 'ds'] = ets_df[date_var]
                ets_df.loc[:, 'y'] = ets_df[target_var]        

                predict_size = predict_n
                ets_train = ets_df.iloc[:-predict_size, :]
                ets_var = ets_df.iloc[-predict_size:, :]

                #auto arima best parameter pick
                ets = Holt(ets_train['y'].values, exponential=True, initialization_method="estimated").fit(
                        smoothing_level=0.8, smoothing_trend=0.2, optimized=False)

                #arima fit 후 ari_var에 예측값 대입
                val_preds = ets.forecast(predict_size)
                ets_var.loc[:, 'pred'] = val_preds

                #train
                train= ets_train[[date_var, target_var]]
                train.loc[:, store_list[0]] = store_var_0
                train.loc[:, store_list[1]] = store_var_1

                train_df = pd.concat([train_df, train], axis=0)

                #valid
                val_preds_df = ets_var[[date_var, target_var, 'pred']]
                val_preds_df.loc[:, store_list[0]] = store_var_0
                val_preds_df.loc[:, store_list[1]] = store_var_1

                val_df = pd.concat([val_df, val_preds_df], axis=0)

                #predict_date 생성 후 예측 predict_df생성
                last_date = ets_df[date_var].iloc[-1:].tolist()[0]

                if unit =='day':
                    predict_date = [last_date + timedelta(days=i) for i in range(1, predict_n+1)] #weeks, days 변경 가능
                elif unit == 'week':
                    predict_date = [last_date + timedelta(days=7*i) for i in range(1, predict_n+1)] #weeks, days 변경 가능
                elif unit == 'month':
                    predict_date = [last_date + timedelta(days=30*i) for i in range(1, predict_n+1)] #weeks, days 변경 가능

                test_df = pd.DataFrame({date_var:predict_date})
                ets = Holt(ets_df['y'].values, exponential=True, initialization_method="estimated").fit(
                        smoothing_level=0.8, smoothing_trend=0.2, optimized=False)

                preds = ets.forecast(predict_size)
                test_df.loc[:, 'pred'] = preds
                test_df.loc[:, store_list[0]] = store_var_0
                test_df.loc[:, store_list[1]] = store_var_1

                pred_df = pd.concat([pred_df, test_df], axis=0)
                
        except:
            self.logger.exception('ets 모델링 도중 문제가 발생하였습니다.')
                           
        self.result_df['train_df']['ets'] = train_df
        self.result_df['val_df']['ets'] = val_df
        self.result_df['pred_df']['ets'] = pred_df        
        
        #모델 평가
        y_test = val_df[target_var].values
        y_pred = val_df['pred'].values
        
        self.model['ets'] = ets
        
        self.score['MAPE']['ets']     = np.round(mape(y_test, y_pred), 3)
        self.score['MAE']['ets']      = np.round(mean_absolute_error(y_test, y_pred), 3)
        self.score['MSE']['ets']      = np.round(mean_squared_error(y_test, y_pred), 3)
        self.score['RMSE']['ets']     = np.round(np.sqrt(mean_squared_error(y_test, y_pred)), 3)
        self.score['상관계수']['ets']  = np.round(r2_score(y_test, y_pred), 3)
        self.score['정확성']['ets']    = np.round(100-mape(y_test, y_pred), 3)

        


    #################### ETS FINISH ####################
    
    #################### FB START####################
    
    def fb_process(self):
        if (self.model_type == 'fb') or (self.model_type == 'auto'): 
            self.fb_fit_predict(self.df, self.target_var, self.date_var, self.store_list, self.unit, self.predict_n, self.HPO)
        
    
    def fb_fit_predict(self, df, target_var, date_var, store_list, unit, predict_n, HPO):
        
        self.logger.info('fbprophet 데이터 준비')
        
        
        if len(store_list) == 1 :
            store_list = ['dummy'] + store_list
            df.loc[:, 'dummy'] = 'dummy'
        
        train_df = pd.DataFrame()
        val_df = pd.DataFrame()
        pred_df = pd.DataFrame()
        
        self.logger.info('fbprophet 모델링 시작')
        try:
            for store_var_0, store_var_1 in df.drop_duplicates(store_list)[store_list].values:
                fb_df = df.loc[(df[store_list[0]]==store_var_0)&(df[store_list[1]]==store_var_1), :]               
                fb_df.loc[:, 'ds'] = fb_df[date_var]
                fb_df.loc[:, 'y'] = fb_df[target_var]        
                fb_df.loc[:, 'cap'] = np.max(fb_df[target_var].values)
                fb_df.loc[:, 'floor'] = 0 #np.min(fb_df[target_var].values)

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
                fb = Prophet(**parameters)
                fb.fit(fb_train[['y','ds','cap','floor']])
                val_preds = fb.predict(fb_var[['ds','cap','floor']])
                val_preds = val_preds[['ds','yhat']]
                
                #컬럼 이름 바꾸기
                val_preds.rename(columns = {'ds'  : date_var, 
                                            'yhat' : 'pred'}, inplace=True)
                
                val_real = fb_var[[date_var, target_var]]
                val_preds_df = pd.merge(val_real, val_preds, on=date_var, how='inner')

                #train
                train = fb_train[[date_var, target_var]]
                train.loc[:, store_list[0]] = store_var_0
                train.loc[:, store_list[1]] = store_var_1
                train_df = pd.concat([train_df, train], axis=0)

                #valid
                val_preds_df.loc[:, store_list[0]] = store_var_0
                val_preds_df.loc[:, store_list[1]] = store_var_1

                val_df = pd.concat([val_df, val_preds_df], axis=0) 
                
                #predict_date 생성 후 예측 predict_df생성
                last_date = fb_df[date_var].iloc[-1:].tolist()[0]

                if unit =='day':
                    predict_date = [last_date + timedelta(days=i) for i in range(1, predict_n+1)] #weeks, days 변경 가능
                elif unit == 'week':
                    predict_date = [last_date + timedelta(days=7*i) for i in range(1, predict_n+1)] #weeks, days 변경 가능
                elif unit == 'month':
                    predict_date = [last_date + timedelta(days=30*i) for i in range(1, predict_n+1)] #weeks, days 변경 가능
                test_df = pd.DataFrame({'ds': predict_date})
                test_df['cap'] = fb_df['cap'].values[0]
                test_df['floor'] = fb_df['floor'].values[0]

                preds = fb.predict(test_df[['ds','cap','floor']])
                preds = preds[['ds','yhat']]
                
                #컬럼 이름 바꾸기
                preds.rename(columns = {'ds' : date_var,
                                        'yhat': 'pred'}, inplace=True)
                             
                preds.loc[:, store_list[0]] = store_var_0
                preds.loc[:, store_list[1]] = store_var_1
                pred_df = pd.concat([pred_df, preds], axis=0)
        
        except:
                self.logger.exception('fbprophet 모델링 도중 문제가 발생하였습니다.')
        
        self.result_df['train_df']['fb'] = train_df
        self.result_df['val_df']['fb'] = val_df
        self.result_df['pred_df']['fb'] = pred_df 
#         train_df.to_csv('result/fb_train_df.csv', index=False)
#         val_df.to_csv('result/fb_val_df.csv', index=False)
#         pred_df.to_csv('result/fb_pred_df.csv', index=False)
        
        y_test = val_df[target_var].values
        y_pred = val_df['pred'].values
        
        self.model['fb'] = fb
        
        self.score['MAPE']['fb']     = np.round(mape(y_test, y_pred), 3)
        self.score['MAE']['fb']      = np.round(mean_absolute_error(y_test, y_pred), 3)
        self.score['MSE']['fb']      = np.round(mean_squared_error(y_test, y_pred), 3)
        self.score['RMSE']['fb']     = np.round(np.sqrt(mean_squared_error(y_test, y_pred)), 3)
        self.score['상관계수']['fb']  = np.round(r2_score(y_test, y_pred), 3)
        self.score['정확성']['fb']    = np.round(100-mape(y_test, y_pred), 3)
  


    #################### FB FINISH #################### 
    
    #################### TFT START #################### 
    def tft_process(self):
        #if (self.model_type == 'tft') or (self.model_type == 'auto'): 
        self.tft_fit_predict(self.df, self.target_var, self.date_var, self.store_list, self.unit, self.predict_n, self.HPO)
            
    
    def tft_fit_predict(self, df, target_var, date_var, store_list, unit, predict_n, HPO):
        
        self.logger.info('tft 데이터 준비')
        
        self.logger.info('tft 데이터 전처리')
        try:
            df[f"log_{target_var}"] = np.log(np.abs(df[target_var]) + 1e-4)
            for store_var in store_list:
                df[f"avg_{target_var}_by_{store_var}"] = df.groupby(["time_idx", store_var], observed=True)[target_var].transform("mean")

            max_prediction_length = predict_n
            max_encoder_length = predict_n * 4
            training_cutoff = df['time_idx'].max() - max_prediction_length

            #매우 중요
            df.sort_values(store_list+[date_var], inplace=True)
        
        except:
            self.logger.exception('tft 데이터 전처리 도중 문제가 발생하였습니다.')
                
        self.logger.info('tft dataloader 세팅')
        try:
            training = TimeSeriesDataSet(
                df[lambda x: x.time_idx <= training_cutoff],
                time_idx="time_idx",
                target=target_var,
                group_ids=store_list,
                min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
                max_encoder_length=max_encoder_length,
                min_prediction_length=1,
                max_prediction_length=max_prediction_length,
                static_categoricals=store_list,
                static_reals=[],
                time_varying_known_categoricals=[unit],
                variable_groups={},  # group of categorical variables can be treated as one variable
                time_varying_known_reals=["time_idx"],
                time_varying_unknown_categoricals=[],
                time_varying_unknown_reals=
                [target_var, f"log_{target_var}"]+ [f"avg_{target_var}_by_{store_var}" for store_var in store_list],
                target_normalizer=GroupNormalizer(
                    groups=store_list, transformation="softplus"
                ),  # use softplus and normalize by group
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
            )

            # create validation set (predict=True) which means to predict the last max_prediction_length points in time
            # for each series
            validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

            # create dataloaders for model
            batch_size = 128  # set this between 32 to 128
            train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
            val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
            
            pl.seed_everything(42)
            trainer = pl.Trainer(
                gpus=0,
                # clipping gradients is a hyperparameter and important to prevent divergance
                # of the gradient for recurrent neural networks
                gradient_clip_val=0.1,
            )


            tft = TemporalFusionTransformer.from_dataset(
                training,
                # not meaningful for finding the learning rate but otherwise very important
                learning_rate=0.03,
                hidden_size=16,  # most important hyperparameter apart from learning rate
                # number of attention heads. Set to up to 4 for large datasets
                attention_head_size=1,
                dropout=0.1,  # between 0.1 and 0.3 are good values
                hidden_continuous_size=8,  # set to <= hidden_size
                output_size=7,  # 7 quantiles by default
                loss=QuantileLoss(),
                # reduce learning rate if no improvement in validation loss after x epochs
                reduce_on_plateau_patience=4,
            )
        except:
            self.logger.exception('tft dataloader 세팅 도중 문제가 생겼습니다.')
        
        self.logger.info('tft learning rate 설정')
        try:
            # find optimal learning rate
            res = trainer.tuner.lr_find(
                tft,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
                max_lr=10.0,
                min_lr=1e-6,
            )
        except:
            self.logger.exception('tft learning rate 설정 도중 문제가 생겼습니다.')
         
        self.logger.info('tft 학습')
        try:
            # configure network and trainer
            early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
            lr_logger = LearningRateMonitor()  # log the learning rate
            logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard
            
            if (self.model_type == 'tft') or (self.model_type == 'auto'): 
                n_epochs = 1
            else:
                n_epochs = 1
                
            trainer = pl.Trainer(
                max_epochs=n_epochs,
                gpus=0,
                enable_model_summary=True,
                gradient_clip_val=0.1,
                limit_train_batches=30,  # coment in for training, running valiation every 30 batches
                # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
                callbacks=[lr_logger, early_stop_callback],
                logger=logger,
            )


            tft = TemporalFusionTransformer.from_dataset(
                training,
                learning_rate=0.03,
                hidden_size=16,
                attention_head_size=1,
                dropout=0.1,
                hidden_continuous_size=8,
                output_size=7,  # 7 quantiles by default
                loss=QuantileLoss(),
                log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
                reduce_on_plateau_patience=4,
            )
            
            trainer.fit(
                tft,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )
            
            best_model_path = trainer.checkpoint_callback.best_model_path
            
            tft_path = 'storage/model/tft.ckpt'
            
            #기존 저장소에서 파일 복사
            shutil.copyfile(best_model_path, tft_path)
            best_tft = TemporalFusionTransformer.load_from_checkpoint(tft_path)
        
        except:
            self.logger.exception('tft 학습 도중 문제가 생겼습니다')
            
        self.logger.info('훈련, 검증 데이터 프레임 생성')
        try: 
            
            train_df = df[lambda x: x.time_idx <= x.time_idx.max() - max_prediction_length]
            train_df = train_df[[date_var, target_var] + store_list].reset_index(drop=True)
            
            val_df = df[lambda x: x.time_idx > x.time_idx.max() - max_prediction_length]
            val_df = val_df[[date_var, target_var] + store_list].reset_index(drop=True)
            val_predictions = best_tft.predict(val_dataloader, mode="prediction", return_x=False)
            val_predictions = pd.DataFrame(val_predictions.reshape(-1,1).numpy(), columns=['pred'])

            val_df = pd.concat([val_df, val_predictions], axis=1)
            
        except:
            self.logger.exception('훈련, 검증 데이터 프레임 생성 도중 문제가 생겼습니다')
            
        self.logger.info('예측 데이터 프레임 생성')
        try:
            
            # select last 24 months from data (max_encoder_length is 24)
            encoder_data = df[lambda x: x.time_idx > x.time_idx.max() - max_encoder_length]

            # select last known data point and create decoder data from it by repeating it and incrementing the month
            # in a real world dataset, we should not just forward fill the covariates but specify them to account
            # for changes in special days and prices (which you absolutely should do but we are too lazy here)
            last_data = df[lambda x: x.time_idx == x.time_idx.max()]

            if unit == 'day':
                decoder_data = pd.concat(
                [last_data.assign(date_var = lambda x: x[date_var] + timedelta(days=i),
                                  time_idx = lambda x: x.time_idx + i) 
                                for i in range(1, max_prediction_length + 1)], ignore_index=True, )
            elif unit == 'week':
                decoder_data = pd.concat(
                [last_data.assign(date_var = lambda x: x[date_var] + timedelta(days=7*i),
                                  time_idx = lambda x: x.time_idx + i) 
                                for i in range(1, max_prediction_length + 1)], ignore_index=True, )
            elif unit == 'month':
                decoder_data = pd.concat([last_data.assign(date_var = lambda x: x[date_var] + pd.offsets.MonthBegin(i),
                                  time_idx = lambda x: x.time_idx + i) 
                                for i in range(1, max_prediction_length + 1)], ignore_index=True, )

            #fake date_var 삽입 필요
            decoder_data.loc[:, date_var] = decoder_data['date_var']
            decoder_data.drop('date_var', axis=1, inplace=True)
            decoder_data.sort_values(store_list+[date_var], inplace=True)

            # add additional features
            if unit == 'day':
                decoder_data[unit] = decoder_data[date_var].dt.day.astype(str).astype("category")
            elif unit == 'week':
                decoder_data[unit] = decoder_data[date_var].dt.isocalendar().week.astype(str).astype("category")  # categories have be strings
            elif unit == 'month':
                decoder_data[unit] = decoder_data[date_var].dt.month.astype(str).astype("category")  # categories have be strings

            # combine encoder and decoder data
            new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)
            #매우 중요
            new_prediction_data.sort_values(store_list, inplace=True)
            
            predictions = best_tft.predict(new_prediction_data, mode="prediction", return_x=False)
            predictions = pd.DataFrame(predictions.reshape(-1,1).numpy(), columns=['pred'])
            
            pred_df = decoder_data[[date_var] + store_list].reset_index(drop=True)
            pred_df = pd.concat([pred_df, predictions], axis=1)
        
        except:
            self.logger.exception('예측 데이터 프레임 생성 도중 문제가 생겼습니다')
        
        #변수 중요도!!
        predictions_for_interpret, _ = best_tft.predict(val_dataloader, mode="raw", return_x=True)
        interpretation = best_tft.interpret_output(predictions_for_interpret, reduction="sum")
        time_variables = pd.DataFrame(interpretation['attention'].numpy(),
                                      columns= ['Time index'],
                                      index = [i-len(interpretation['attention']) for i in range(len(interpretation['attention']))])
        static_variables = pd.DataFrame(interpretation['static_variables'].numpy()/interpretation['static_variables'].numpy().sum(), 
                                        index = [best_tft.static_variables], 
                                        columns=['정적공변량 변수중요도'])
        encoder_variables = pd.DataFrame(interpretation['encoder_variables'].numpy()/interpretation['encoder_variables'].numpy().sum(),
                                         index = [best_tft.encoder_variables], 
                                         columns=['인코더 변수중요도'])
        
        self.vi['time'] = time_variables
        self.vi['static'] = static_variables
        self.vi['encoder'] = encoder_variables               
                        
        self.result_df['train_df']['tft'] = train_df
        self.result_df['val_df']['tft'] = val_df
        self.result_df['pred_df']['tft'] = pred_df 
#         train_df.to_csv('result/tft_train_df.csv', index=False)
#         val_df.to_csv('result/tft_val_df.csv', index=False)
#         pred_df.to_csv('result/tft_pred_df.csv', index=False)
        
        y_test = val_df[target_var].values
        y_pred = val_df['pred'].values
        
        self.model['tft'] = best_model_path
        
        self.score['MAPE']['tft']     = np.round(mape(y_test, y_pred), 3)
        self.score['MAE']['tft']      = np.round(mean_absolute_error(y_test, y_pred), 3)
        self.score['MSE']['tft']      = np.round(mean_squared_error(y_test, y_pred), 3)
        self.score['RMSE']['tft']     = np.round(np.sqrt(mean_squared_error(y_test, y_pred)), 3)
        self.score['상관계수']['tft']  = np.round(r2_score(y_test, y_pred), 3)
        self.score['정확성']['tft']    = np.round(100-mape(y_test, y_pred), 3)
     

    #################### TFT FINISH #################### 
            
       
    def get_best_model(self):
        
        self.logger.info('Auto ML 가동')
        self.logger.info(f'automl_score:{self.score}')
        try:
            best_model_name = min(self.score['MAE'], key=self.score['MAE'].get) 
            best_model = self.model[best_model_name]
            
            
            
            self.logger.info(f'best_model_name: {best_model_name}')
                                  
        except:
            self.logger.exception('best 모델 선정에 실패했습니다')                                                                 
        
        
        self.logger.info('best 모델 저장')
        try:
            # 모델 저장
            self.logger.info(best_model)
            if 'tft' in best_model_name:
                joblib.dump(best_model, best_model)
                #checkpoint 불러와야 함
            else :
                joblib.dump(best_model, 'storage/model/best_model.pkl')
        
            #모델 이름 저장
            model_name = {'best_model_name' : best_model_name}
            with open('storage/model/model_name.json', 'w') as f:
                json.dump(model_name, f)

        except:
            self.logger.exception('best 모델 저장에 실패했습니다')                                                                 
        
        return best_model_name, best_model #, best_test
    
    #1. 분석 리포트
    def make_report(self, target_var, date_var, unit, model_type, hpo, start_time):
        
        self.logger.info('학습 결과를 위한 결과물 생성')
        try:
            report = pd.DataFrame({'상태' : '완료됨',
                                  '모델 ID' : ['model_id'],
                                  '생성 시각': [start_time.strftime('%Y-%m-%d %H:%M:%S')],
                                  '학습 시간' : [datetime.datetime.now()-start_time],
                                   '데이터셋 ID' : 'dataset_id',
                                   '타겟 변수' : target_var,
                                   '날짜 변수' : date_var,
                                   '시간 단위' : unit,
                                   '알고리즘' : model_type, 
                                   '목표' : '테이블 형식 시계열',
                                   '최적화 목표' : 'MAPE',
                                   'HPO 여부' : hpo})
            
            report = report.T
        
        except:
            self.logger.exception('학습 결과를 위한 결과물 생성 실패했습니다')
            
        return report
    
    #2. 학습 결과 비교 화면
    #3. 변수 중요도와 시간 중요도
    def get_eval(self, best_model_name):
                 
        self.logger.info('best 모델 및 전체 모델 검증치 추출')
        try:
            test_score = pd.DataFrame({'MAPE'  : [self.score['MAPE'][best_model_name]],
                                      'MAE'    : [self.score['MAE'][best_model_name]],     
                                      'MSE'    : [self.score['MSE'][best_model_name]], 
                                      'RMSE'   : [self.score['RMSE'][best_model_name]],
                                      '상관계수': [self.score['상관계수'][best_model_name]],
                                      '정확성'  : [self.score['정확성'][best_model_name]]
                                      })
                 
            valid_score = pd.DataFrame(self.score)
                 
            
            best_train_df = self.result_df['train_df'][best_model_name]
            best_val_df = self.result_df['val_df'][best_model_name]
            best_pred_df = self.result_df['pred_df'][best_model_name]
            
            best_train_df.to_csv('result/best_train_df.csv', index=False)
            best_val_df.to_csv('result/best_val_df.csv', index=False)
            best_pred_df.to_csv('result/best_pred_df.csv', index=False)
            
            vi_time = self.vi['time'] 
            vi_static = self.vi['static'] 
            vi_encoder = self.vi['encoder'] 
            
            vi_time.to_csv('result/vi_time.csv', index=False)
            vi_static.to_csv('result/vi_static.csv', index=False)
            vi_encoder.to_csv('result/vi_encoder.csv', index=False)
         
        except:
            self.logger.exception('학습 결과를 위한 결과물 생성 실패했습니다')
                                      
        return test_score, valid_score, best_train_df, best_val_df, best_pred_df, vi_time, vi_static, vi_encoder   