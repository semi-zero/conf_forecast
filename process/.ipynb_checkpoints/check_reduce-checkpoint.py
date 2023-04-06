import pandas as pd
import logging
import numpy as np

class Data_check_reduce:
    def __init__(self, log_name, data, target_var, date_var, store_list, predict_n):
        self.logger = logging.getLogger(log_name)
        
        self.df = data
        self.target_var = target_var
        self.date_var = date_var
        self.store_list = store_list
        self.n = predict_n
        
        self.logger = logging.getLogger(log_name)
        
        #데이터 check가 통과되지 못하면 False로 변경
        self.check = True
        
        #데이터 check
        self.data_check()
        
        
    def data_check(self):
        
        self.logger.info('데이터 정합성 확인 절차 시작')        
        
        
        if self.df[self.target_var].isnull().sum() != 0:
            self.logger.info('타겟 변수에 결측치가 포함되어 있습니다')
            self.check = False
        
        #if min(self.df[self.store_list].value_counts()) != max(self.df[self.store_list].value_counts()) :
        #    self.logger.info('시계열 식별 변수별 데이터의 개수가 다릅니다')
        #    self.check = False
        
        if min(self.df[self.store_list].value_counts()) <= (self.n * 4):
            self.logger.info('예측하고자 하는 기간의 수가 훈련 데이터에 비해 깁니다.')
            self.check = False
            
        try:
            self.df[self.date_var] = pd.to_datetime(self.df[self.date_var],infer_datetime_format = True, utc = True).astype('datetime64[ns]')
                
        except:
            
            try:
                
                self.df[self.date_var] = self.df[self.date_var].apply(lambda x : datetime.datetime.strptime(str(x), '%Y%m%d'))
            
            except:
                
                self.logger.exception('시간 변수는 시간타입으로 변환 가능해야 합니다')
                self.check = False
                    
        
        self.logger.info('데이터 정합성 확인 절차 종료')
        self.logger.info(f'데이터 정합성 확인 절차 결과 : {self.check}')    
    
    
