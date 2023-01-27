from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics


import optuna
from optuna import Trial
from optuna.samplers import TPESampler

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class HyperOptimization:
    
    def __init__(self, train, valid, model):
     
        self.train = train
        self.valid = valid
        self.model = model

        self.obj = {'fb' : self.fb_objective}

        sampler = TPESampler(seed=42)
        study = optuna.create_study(study_name="parameter_opt", direction="minimize", sampler=sampler,)
        study.optimize(self.obj[self.model], n_trials=1)
        self.best_params = study.best_params
    
    #n_estimator 10
        
        
    def fb_objective(self, trial):
        params = {
        'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.005, 5),
        'changepoint_range': trial.suggest_float('changepoint_range', 0.8, 0.9),
        'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.1, 10),
        'holidays_prior_scale': trial.suggest_float('holidays_prior_scale', 0.1, 10),
        'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['multiplicative', 'additive']),
        #'growth': trial.suggest_categorical('growth', ['linear', 'logistic']),
        'weekly_seasonality': trial.suggest_int('weekly_seasonality', 5, 10),
        'yearly_seasonality': trial.suggest_int('yearly_seasonality', 1, 20)
        }

        m = Prophet(**params)
        m.fit(self.train[['y','ds','cap','floor']])
        preds = m.predict(self.valid[['ds','cap','floor']])

        mae_score = mean_absolute_error(self.valid['y'], preds['yhat'])

        return mae_score