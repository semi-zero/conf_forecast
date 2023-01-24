import pandas as pd
import logging
import numpy as np

class Data_load:
    def __init__(self, path, log_name):
        self.path = path #데이터 위치 경로 입력
        self.logger = logging.getLogger(log_name)

    def read_data(self):

        self.logger.info('csv 데이터 불러오기')
        self.logger.info(f'{self.path}')
        try:
            df = pd.read_csv(self.path) 
            var_list = df.columns.tolist() #전체 변수리스트 추출
            num_var = df.select_dtypes(include='float').columns.tolist() + df.select_dtypes(include='int').columns.tolist() #수치형 변수 추출
            obj_var = [x for x in df.columns if x not in num_var] #문자형 변수 추출
        
        except: 
            self.logger.error('csv 데이터 불러오기를 실패했습니다')
        
        df = self.reduce_mem_usage(df)
        
        return df, var_list, num_var, obj_var
    
    #데이터 메모리 줄이기
    def reduce_mem_usage(self, df):
        """ 
        iterate through all the columns of a dataframe and 
        modify the data type to reduce memory usage.        
        """
        start_mem = df.memory_usage().sum() / 1024**2
        self.logger.info(f'데이터 구성: {df.shape[0]} 행, {df.shape[1]}열')
        self.logger.info(f'Memory usage of dataframe is {start_mem:.2f}MB')
    
        for col in df.columns:
            col_type = df[col].dtype
        
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max <\
                    np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max <\
                    np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max <\
                    np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max <\
                    np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                elif str(col_type)[:5] == 'float':
                    if c_min > np.finfo(np.float16).min and c_max <\
                    np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max <\
                    np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
                else:
                    pass
            else:
                df[col] = df[col].astype('category')
        end_mem = df.memory_usage().sum() / 1024**2
        self.logger.info(f'Memory usage after optimization is: {end_mem:.2f}MB')
        self.logger.info(f'Decreased by {100*((start_mem - end_mem)/start_mem):.1f}%')
    
        return df