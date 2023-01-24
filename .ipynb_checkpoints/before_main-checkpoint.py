import pandas as pd
from lightgbm import plot_importance
from lightgbm import LGBMClassifier


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
from datetime import timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima
from statsmodels.tsa.stattools import adfuller

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

import argparse

# 1. parser 객체 생성
parser = argparse.ArgumentParser(description='Click & Select')
# 2. 사용할 인수 등록,  이름/타입/help
parser.add_argument('-pth', '--PATH',  type=str, help='Path of Data')        # 데이터 위치 경로
parser.add_argument('-target', '--target', type=str, help='Target vairable') # 타겟 변수 
parser.add_argument('-date', '--date', type=str)                             # 날짜 변수
parser.add_argument('-store', '--store', type=str)                           # 지점 변수
parser.add_argument('-types', '--types', type=str)                           # validate vs predict
parser.add_argument('-n', '--predict_n', type=int)                           # 예측해야하는 기간
parser.add_argument('-ps', '--predict_store', nargs='+', type=int)           # 예측하고자 하는 지점 번호  
parser.add_argument('-ap', '--anomaly_per', type=int)                        # 전처리 이상치 비율
parser.add_argument('-na', '--na', type=bool)                                # 결측치 처리 여부 
parser.add_argument('-out', '--outlier', type=bool)                          # 이상치 처리 여부

args = parser.parse_args()


class Data_load:
    def __init__(self, path):
        self.path = path #데이터 위치 경로 입력
    
    
    # 데이터 불러오기
    def read_data(self):
        df = pd.read_csv(self.path) 
        var_list = df.columns.tolist() #전체 변수리스트 추출
        num_var = df.select_dtypes(include='float').columns.tolist() + df.select_dtypes(include='int').columns.tolist() #수치형 변수 추출
        obj_var = [x for x in df.columns if x not in num_var] #문자형 변수 추출
        return df, var_list, num_var, obj_var
    


#전처리 class
class Preprocessing:
    def __init__(self, data, var_list, num_var, obj_var, target_var, date_var, store_var, anomaly_per, na=False, outlier=False, tree_model=False):
        self.df = data                                     # 데이터
        self.target = target_var                           # 타겟 변수
        self.date = date_var                               # 시간 변수
        self.store = store_var                             # 지점 변수
        self.var_list = var_list                           # 전체 변수 리스트
        self.num_var = num_var                             # 수치형 변수 리스트
        self.obj_var = obj_var                             # 문자형 변수 리스트
        self._anomaly_ratio = int(anomaly_per)             # 지정 이상치 범위
        self._anomaly_percentage = int(anomaly_per) / 100
        
        self.na_pre = na                                   #결측치 제거 여부 (제거=True, 보간=False )
        self.outlier_pre = outlier                         #이상치 처리 여부
        self.stand_pre = tree_model                        #정규화 처리 여부 => 트리 모델의 경우 생략 가능
        
        
        self.df = self.na_preprocess(self.df, self._anomaly_ratio, self.na_pre)
        
        if self.outlier_pre:
            self.df = self.outlier_preprocess(self.df, self.num_var)
        
        #시간 변수 분리
        if self.date in self.obj_var:
            self.obj_var.remove(self.date)
        else: self.num_var.remove(self.date)
            
        #지점 변수 분리
        if self.target in self.obj_var:
            self.obj_var.remove(self.target)
        else: self.num_var.remove(self.target)
        
        #target 변수 분리
        if self.store in self.obj_var:
            self.obj_var.remove(self.store)
        else: self.num_var.remove(self.store)
    
        
        if self.stand_pre is False:
            self.df = self.standardize(self.df, self.num_var)
        
        self.df = self.label_encoder(self.df, self.obj_var)
        
        self.df = self.ts_preprocess(self.df, self.target, self.date, self.store)
        
    # 결측치 확인 및 처리
    def na_preprocess(self, df, anomaly_per, na):
        
        if na is True:
            
            #Column별 결측치 n% 이상 있을 경우 제외
            remove_v1 = round(df.isnull().sum() / len(df)*100, 2)
            tmp_df = df[remove_v1[remove_v1 < anomaly_per].index]
        
            #Row별 결측치 n% 이상 있을 경우 제외
            idx1 = len(tmp_df.columns) * 0.7
            tmp_df.dropna(thresh=idx1, axis=0)
            
        else:

            #보강
            tmp_df = df.fillna(0)
            
        print('결측치 처리')
        return tmp_df
        
        
    # 이상치 제거
    def outlier_preprocess(self, df, num_var):
        num_data = df.loc[:, num_var]
        
        #IQR 기준
        quartile_1 = num_data.quantile(0.25)
        quartile_3 = num_data.quantile(0.75)
        IQR = quartile_3 - quartile_1

        condition = (num_data < (quartile_1 - 1.5 * IQR)) | (num_data > (quartile_3 + 1.5 * IQR)) # 1.5 수치가 바뀌어야함
        condition = condition.any(axis=1)
        search_df = df[condition]
        print('이상치 처리')
        return df.drop(search_df.index, axis=0)
        
    
        
    #트리 모델이 아닐 경우 표준화 진행
    def standardize(self, df, num_var):
        if len(num_var) > 0:
            num_data = df.loc[:, num_var]
            non_num_data = df.drop(set(num_var), axis=1)
        
            #표준화
            std_scaler = StandardScaler()
            fitted = std_scaler.fit(num_data)
            output = std_scaler.transform(num_data)
            num_data = pd.DataFrame(output, columns = num_data.columns, index=list(num_data.index.values))
        
            tmp_df = pd.concat([non_num_data, num_data], axis=1)
            print('표준화')
        else: tmd_df = df
        
        return tmp_df
        
    
    #문자형 변수를 수치형으로 변환
    def label_encoder(self, df, obj_var):
        if len(obj_var) > 0:
            obj_data = df.loc[:, obj_var]
            non_obj_data = df.drop(set(obj_var), axis=1)

            #인코딩
            obj_output = pd.DataFrame()
            for obj_col in obj_var:
                lb_encoder = LabelEncoder()
                output = lb_encoder.fit_transform(obj_data.loc[:, obj_col])
                output = pd.DataFrame(output, index = list(obj_data.index.values))
                obj_output = pd.concat([obj_output, output], axis=1)
            obj_output.columns = obj_var
            tmp_df = pd.concat([obj_output, non_obj_data], axis=1)
            print('수치형 변환')
        
        else: tmp_df = df
            
        return tmp_df
    
    
    #시계열 처리를 위한 변수 변환
    def ts_preprocess(self, df, target_var, date_var, store_var):
        tmp_df = pd.DataFrame()
        for store_number in df[store_var].unique():
            store_df = df[df[store_var] == store_number ]
            store_df['mean14'] = round(store_df[target_var].rolling(window=14, min_periods=1).mean(),2)
            tmp_df = pd.concat([tmp_df, store_df], axis=0)
        tmp_df[date_var] = tmp_df[date_var].apply(lambda x : pd.to_datetime(str(x)))
        tmp_df = tmp_df.sort_values(by=[store_var, date_var])
        return tmp_df
        

#ARIMA 모델링
class Arima_modeling:
    def __init__(self, df, target_var, date_var, store_var, types, n, store_num):
        self.df = df                                           # 데이터
        self.target_var = target_var                           # 타겟 변수
        self.date_var = date_var                               # 시간 변수
        self.store_var = store_var                             # 지점 변수
        self.types = types                                     # 예측 또는 검증
        self.predict_n = n                                     # 예측 기간
        self.store_num = store_num                             # 예측하려는 지점
        
        
        #훈련 및 검증 용 데이터 생성
        self.extra_df, self.train, self.test, self.y_1diff, self.y_log, self.y_log_1diff, self.log_moving_avg, self.log_moving_avg_diff = self.make_arima_ds(self.df
                                                                                                                                              , self.target_var
                                                                                                                                              , self.date_var
                                                                                                                                              , self.store_var
                                                                                                                                              , self.predict_n
                                                                                                                                              , self.store_num)
        
        #훈련 모델 생성
        self.model, self.model_diff, self.model_log_diff, self.model_log_mm = self.arima_fit(self.train
                                                                                           , self.test
                                                                                           , self.y_1diff
                                                                                           , self.y_log
                                                                                           , self.y_log_1diff
                                                                                           , self.log_moving_avg
                                                                                           , self.log_moving_avg_diff)
        
        if self.types == 'validate' : 
            self.result_df = self.arima_validate(self.test, self.model)
        else: 
            self.result_df = self.arima_predict(self.extra_df, self.model_diff, self.model_log_diff, self.model_log_mm)
            
            
    def make_arima_ds(self, df, target_var, date_var, store_var, n, store_num):
        #준비
        pre_df = df[df[store_var]==store_num]
        pre_df = pre_df[[date_var, target_var]]
        
        last_date = pre_df[date_var].iloc[-1:].tolist()[0]
        extra_date = [last_date + timedelta(weeks=i) for i in range(1, n+1)] #weeks, days 변경 가능
        extra_df = pd.DataFrame({date_var: extra_date}, columns = [date_var, target_var])
        extra_df = extra_df.set_index(date_var)
        
        pre_df = pd.concat([pre_df, extra_df], ignore_index=True)
        pre_df = pre_df.set_index(date_var)
        data = pre_df[:-n]
        
        #제작
        y = data[target_var].values
        y_1diff  = data.diff().dropna()                                                                      # 1차 차분
        y_log = [np.log(num) if num!= 0 else 0 for num in y]                                                 # log 변환
        y_log_1diff = data[target_var].apply(lambda x: np.log(x) if x != 0 else 0).diff().dropna()           # log + 1차 차분
        log_moving_avg = pd.Series(y_log).rolling(window=4, center=False).mean()                            
        log_moving_avg_diff = (y_log - log_moving_avg).dropna()                                              # log + moving average n일
        
        #train, test
        train = data[:-n]
        test = data[-n:]
        
        return extra_df, train, test, y_1diff, y_log, y_log_1diff, log_moving_avg, log_moving_avg_diff
    
    
    def arima_fit(self, train, test, y_1diff, y_log, y_log_1diff, log_moving_avg, log_moving_avg_diff):
        
        model = auto_arima(train, trace=True, error_action = 'ignore', start_p=1, start_q=1, d=1, max_p=5, max_q=5, 
                  suppress_warning=True, stepwise=True, seasonal=True)
        model.fit(train)
        
        # 1차 차분
        model_diff = auto_arima(y_1diff, trace=True, error_action = 'ignore', start_p=1, start_q=1, d=1, max_p=5, max_q=5, 
                  suppress_warning=True, stepwise=True, seasonal=True)
        model_diff.fit(train)
        
        # log + 1차 차분
        model_log_diff = auto_arima(y_log_1diff, trace=True, error_action = 'ignore', start_p=1, start_q=1, d=1, max_p=5, max_q=5, 
                  suppress_warning=True, stepwise=True, seasonal=True)
        model_log_diff.fit(train)
        
        # log + moving average n일
        model_log_mm = auto_arima(log_moving_avg_diff.values, trace=True, error_action = 'ignore', start_p=1, start_q=1, d=1, max_p=5, max_q=5, 
                  suppress_warning=True, stepwise=True, seasonal=True)
        model_log_mm.fit(train)
        
        return model, model_diff, model_log_diff, model_log_mm

    
    def arima_validate(self, test, model):
        n = len(test)
        result_df = test 
        result_df['pred'] = model.predict(n_periods=n).values.round()
        
        return result_df
    
    def arima_predict(self, extra_df, model_diff, model_log_diff, model_log_mm):
        n = len(extra_df)

        # 1차 차분
        pred_diff = model_diff.predict(n_periods=n) 
        pred_diff = pd.DataFrame(pred_diff.round(), columns={'pred_diff'})
        pred_diff.index = extra_df.index
                      
        # # log + 1차 차분
        pred_log_diff = model_log_diff.predict(n_periods=n)
        pred_log_diff = pd.DataFrame(pred_log_diff.round(), columns={'pred_log_diff'})
        pred_log_diff.index = extra_df.index
        
        # log + moving average n일
        pred_log_mm = model_log_mm.predict(n_periods=n)
        pred_log_mm = pd.DataFrame(pred_log_mm.round(), columns={'pred_log_mm'})
        pred_log_mm.index = extra_df.index
        
        return pd.concat([pred_diff, pred_log_diff, pred_log_mm], axis=1)


# 여러 개의 지점을 예측할 경우와 한 개의 지점을 예측할 경우를 나눠 결과값 생성
def make_result(df, target_var, date_var, store_var, types, n, store):
    if type(store) == list :
        result = pd.DataFrame()
        train = pd.DataFrame()
        for i in store:
            Am = Arima_modeling(df, target_var, date_var, store_var, types, n, store_num=i)
            #tmp_df = pd.DataFrame()
            tmp_result = Am.result_df
            tmp_result[store_var] = i
            result = pd.concat([result, tmp_result], axis=0)
            
            tmp_train = Am.train
            tmp_train[store_var] = i
            train = pd.concat([train, tmp_train], axis=0)
            
    else:
        Am =  Arima_modeling(df, target_var, date_var, store_var, types, n, store_num=store)
        result = Am.result_df
        train = Am.train
        
        result['Store'] = store
        train['Store'] = store
    
    return result, train


if __name__ == "__main__":
    data, var_list, num_var, obj_var = Data_load(args.PATH).read_data()
    df = Preprocessing(data, var_list, num_var, obj_var, target_var=args.target, date_var=args.date, store_var=args.store, anomaly_per=args.anomaly_per, na=args.na, outlier=args.outlier, tree_model=False).df
    result, train = make_result(df, target_var=args.target, date_var=args.date, store_var=args.store, types=args.types, n=args.predict_n, store=args.predict_store)


    # 입력 예시
    # python main.py -pth Walmart.csv -target Weekly_Sales -date Date -store Store -types predict -n 7 -ps 1 2 3 4 5 -ap 10 -na True -out False 
    # python main.py -pth Walmart.csv -target Weekly_Sales -date Date -store Store -types predict -n 7 -ps 1 -ap 10 -na True -out False 
