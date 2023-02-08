#automl_forecast
import argparse
import logging
import pandas as pd
import random
import os
import numpy as np
from loggers import logger
from process import input_data, preprocess, modeling

# 1. parser 객체 생성
parser = argparse.ArgumentParser(description='Click & Select')

# 2. 사용할 인수 등록,  이름/타입/help
parser.add_argument('-pth', '--PATH',  type=str, help='Path of Data')
parser.add_argument('-target', '--target', type=str, help='Target vairable')
parser.add_argument('-date', '--date', type=str, help='ID variable')
parser.add_argument('-store', '--store', nargs='+', type=str, help='Store variable')
parser.add_argument('-unit', '--unit', type=str, help='time unit')           
parser.add_argument('-n', '--predict_n', type=int)
parser.add_argument('-model_type', '--model_type', type=str, help='model type')
parser.add_argument('-hpo', '--HPO', action='store_true', help='hyperparameter optimize bool')


args = parser.parse_args()

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)  # type: ignore
    # torch.backends.cudnn.deterministic = True  # type: ignore
    # torch.backends.cudnn.benchmark = True  # type: ignore

def set_logger(log_name):
    log_obj = logger.AutoMLLog(log_name)
    log_obj.set_handler('automl_process')
    log_obj.set_formats()
    auto_logger = log_obj.addOn()
    
    auto_logger.info('logger 세팅')
        

if __name__ == "__main__":
    seed_everything()
    log_name = 'automl_forecast'
    set_logger(log_name)
    data, var_list, num_var, obj_var = input_data.Data_load(args.PATH, log_name).read_data()
    df = preprocess.Preprocessing(log_name, data, var_list, num_var, obj_var, target_var=args.target, date_var= args.date, store_list=args.store, unit=args.unit).df
    mm = modeling.Modeling(log_name, df, target_var=args.target, date_var= args.date, store_list=args.store, unit=args.unit, predict_n = args.predict_n, model_type=args.model_type, HPO=args.HPO)
    
    
    # 입력 예시
    # python main.py -pth storage/Walmart.csv -target Weekly_Sales -date Date -store Store -unit week -n 7 -hpo -model_type auto
    # python main.py -pth storage/Walmart.csv -target Weekly_Sales -date Date -store Store -unit week -n 7 -model_type auto
    # python main.py -pth storage/stallion.csv -target volume -date date -store sku agency -unit month -n 7 -hpo -model_type auto
    # python main.py -pth storage/stallion.csv -target volume -date date -store sku agency -unit month -n 7 -model_type auto
    # python main.py -pth storage/demand_forecast_dataset.csv -target sale_qty -date sale_dy -store str_cd prod_cd -unit day -n 7 -model_type auto
    # python main.py -pth storage/super_dataset_new.csv -target Weekly_Sales -date Date -store Store -unit week -n 14 -model_type auto