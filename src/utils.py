import os
import pickle
import sys 

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import StackingRegressor

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
    except Exception as e:
        raise CustomException(e, sys.exc_info())
    
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        results = []
        model_score = []
        
        for model_name, model in models.items():
            model_params = params.get(model_name, {})
            
            # find the best hyperparameters
            grid_search = GridSearchCV(model, model_params, cv=5, scoring='r2')
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            
            # evaluate the model
            model.set_params(**best_params)
            model.fit(X_train, y_train)
            
            r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2').mean()
            rmse = np.sqrt(-cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error').mean())
            mae = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error').mean()
            
            # evaluate the model on the test set
            y_pred = model.predict(X_test)
            
            # calculate the final metrics
            final_r2 = r2_score(y_test, y_pred)
            final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            final_mae = mean_absolute_error(y_test, y_pred)
            
            results.append({
            'model': model_name,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'final_r2': final_r2,
            'final_rmse': final_rmse,
            'final_mae': final_mae
            })
            
            model_score.append((model_name, final_r2))
        
        df_results = pd.DataFrame(results)
        
        # select top 5 models base on the r2 score
        model_score = sorted(model_score, key=lambda x: x[1], reverse=True)
        top_models = model_score[:5]
        
        estimators = [(name, models[name]) for name, _ in top_models]
        # use stacking models to ensemble the top 5 models
        stacking_models = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
        stacking_models.fit(X_train, y_train)
        
        r2 = cross_val_score(stacking_models, X_train, y_train, cv=5, scoring='r2').mean()
        rmse = np.sqrt(-cross_val_score(stacking_models, X_train, y_train, cv=5, scoring='neg_mean_squared_error').mean())
        mae = cross_val_score(stacking_models, X_train, y_train, cv=5, scoring='neg_mean_absolute_error').mean()
        
        y_pred = stacking_models.predict(X_test)
        final_r2 = r2_score(y_test, y_pred)
        final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        final_mae = mean_absolute_error(y_test, y_pred)
        
        
        stacking_results = pd.DataFrame([{
            'model': 'StackingRegressor',
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'final_r2': final_r2,
            'final_rmse': final_rmse,
            'final_mae': final_mae
        }])
        return stacking_results, stacking_models
    
    except Exception as e:
        raise CustomException(e, sys.exc_info())
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise CustomException(e, sys.exc_info())